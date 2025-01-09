#!/bin/bash

cd $HOME/0/source/HeCBench || exit 1

# disable coredumps
ulimit -c 0

# Which runtime to test
HIP=1
SYCL=0

# Backends to use (both SYCL & chipStar)
# only OpenCL is supported, Level0 does not support CPUs
#OPENCL=1

# test repeats (excluding warmup)
REPEATS=4

###############################

# enable/disable fast math flag
FASTMATH=0
if [ ! -z "$1" ]; then
  FASTMATH=$1
fi

if [ "$FASTMATH" != "0" ]; then
  MATH=fast
else
  MATH=strict
fi

HIP_FLAGS="-ffp-model=${MATH}"
SYCL_FLAGS="-fp-model=${MATH} -fno-sycl-instrument-device-code"

echo "using math flag: ${MATH}"

###############################

# Pocl or Intel CPU OpenCL
POCL=0
if [ ! -z "$2" ]; then
  POCL=$2
fi
echo "using PoCL : ${POCL}"

if [ "$POCL" == "1" ]; then

  CHIP_BDIR=/home/michal/0/build/b_chip_18_release_pocl
  CHIP_INST=/home/michal/0/install/chip_18_release_pocl
  export POCL="_pocl"

  #export POCL_AFFINITY=1
  #export POCL_MAX_WORK_GROUP_SIZE=2048
  #export POCL_CPU_MAX_CU_COUNT=50
  export POCL_CACHE_DIR=/tmp/POCL_CACHE_$RANDOM
  export POCL_BUILDING=1
  # defaults to 1:
  #export POCL_TBB_DEV_PER_NUMA_NODE=0

  #export POCL_DEVICES=basic
  #export OMP_SCHEDULE=auto
  #### export OMP_PROC_BIND=close
  #### export OMP_PLACES="sockets(1)"

  # TRY THESE
  #export POCL_DEVICES=cpu-omp
  #export OMP_SCHEDULE=auto
  #export OMP_NUM_THREADS=104

  #export OMP_DISPLAY_ENV=TRUE
  #unset OMP_DISPLAY_ENV

else

  CHIP_BDIR=/home/michal/0/build/b_chip_18_release_intel
  CHIP_INST=/home/michal/0/install/chip_18_release_intel

  export POCL="_intel"

  # Intel Opencl for CPU needs oneapi
  source /opt/intel/oneapi/setvars.sh
fi

echo "CHIP bdir : ${CHIP_BDIR}"
echo "CHIP inst : ${CHIP_INST}"

###############################

# test result file suffix
SUFFIX="${REPEATS}x_${MATH}_${POCL}"
echo "SUFFIX: ${SUFFIX}"

# clean everything in HeCBench repo (usually not necessary, --clean option enough)
#git clean -f -d -x
#ln -s /nfs/site/home/mbabej/data .

#############################################

echo "@@@@@@@@@@@@@@@@@@@@@@@@@ Running tests on HIP -> OPENCL :"
clinfo -l
sleep 4

if [ $HIP -ne 0 ]; then

# export CHIP_DUMP_SPIRV=1

# put HIPCC into path, used by python script
export PATH=${CHIP_INST}/bin:$PATH
which hipcc
sleep 4

#skip uninit
export CHIP_SKIP_UNINIT=1
# disable lazy compilation
export CHIP_LAZY_JIT=0

export CHIP_DEVICE_TYPE=cpu
export CHIP_LOGLEVEL=crit

export CHIP_BE=opencl
#export CHIP_PLATFORM=0

$CHIP_BDIR/hipInfo

sleep 5

./scripts/autohecbench.py --clean --warmup true --repeat ${REPEATS} --extra-compile-flags="${HIP_FLAGS}" -o ../test_FULL_${SUFFIX}_hip_oclBE.csv hip

fi

#############################################

echo "@@@@@@@@@@@@@@@@@@@@@@@@@ Running tests on SYCL -> OpenCL:"
sycl-ls
sleep 4

if [ $SYCL -ne 0 ]; then

# export SYCL_DUMP_IMAGES=1

# enable SYCL cache in a random directory in TMP
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=/tmp/SYCL_CACHE_${RANDOM}

echo "RUNNING OPENCL tests"
#export SYCL_DEVICE_FILTER=opencl:cpu
export ONEAPI_DEVICE_SELECTOR=opencl:cpu

sycl-ls

sleep 4

./scripts/autohecbench.py --verbose  --warmup true --repeat ${REPEATS} --extra-compile-flags="${SYCL_FLAGS}" -o ../test_FULL_${SUFFIX}_sycl_oclBE.csv --sycl-type opencl sycl

fi

#############################################
