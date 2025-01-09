#!/bin/bash

cd $HOME/0/source/HeCBench || exit 1

# disable coredumps
ulimit -c 0

# required hack
export LD_LIBRARY_PATH=/home/LLVM_17/lib

# ORTCE environment
#source /opt/intel/oneapi/setvars.sh
#source /opt/intel/oneapi/PVC_setup.sh
#source /etc/profile.d/modules.sh

# load specific version of GPU driver
# current default = 775.20
#module load intel-comp-rt/agama-ci-devel/735
#module list

# Which runtime to test
HIP=1
SYCL=0

# Backends to use (both SYCL & chipStar)
LEVEL0=0
OPENCL=1

# test repeats (excluding warmup)
REPEATS=4

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
# cat /etc/os-release
sleep 4

SUFFIX=0
# chipstar build dir
CHIP_BDIR=$HOME/0/build/b_chip_17_release
# chipstar install dir
CHIP_INST=$HOME/0/install
# test result file suffix
SUFFIX=${REPEATS}_${MATH}

echo "CHIP bdir : ${CHIP_BDIR}"
echo "CHIP inst : ${CHIP_INST}"
echo "SUFFIX: ${SUFFIX}"

# clean everything in HeCBench repo (usually not necessary, --clean option enough)
#git clean -f -d -x
#ln -s /nfs/site/home/mbabej/data .

echo "@@@@@@@@@@@@@@@@@@@@@@@@@ OPENCL DEVICES:"
clinfo -l
sleep 2

#############################################

if [ $HIP -ne 0 ]; then

# export CHIP_DUMP_SPIRV=1
# minkowski-hip 
#export HIP_TESTS=(nlll-hip softmax-hip mrc-hip tsa-hip chemv-hip)
export HIP_TESTS=(fft-hip)


# put HIPCC into path, used by python script
export PATH=${CHIP_INST}/bin:$PATH
which hipcc
sleep 4

export CHIP_L0_COLLECT_EVENTS_TIMEOUT=5
export CHIP_L0_IMM_CMD_LISTS=1

#skip uninit
export CHIP_SKIP_UNINIT=1
# disable lazy compilation
export CHIP_LAZY_JIT=0

export CHIP_DEVICE_TYPE=gpu
export CHIP_LOGLEVEL=warn

if [ $OPENCL -ne 0 ]; then
  export CHIP_BE=opencl
  export CHIP_PLATFORM=0
  $CHIP_BDIR/hipInfo
  sleep 5
  ./scripts/autohecbench.py --clean --warmup true --repeat ${REPEATS} --extra-compile-flags="${HIP_FLAGS}" -o ../test_FULL_${SUFFIX}_hip_oclBE.csv "${HIP_TESTS[@]}"
fi


if [ $LEVEL0 -ne 0 ]; then
  export CHIP_BE=level0
  unset CHIP_PLATFORM
  ./scripts/autohecbench.py --clean --warmup true --repeat ${REPEATS} --extra-compile-flags="${HIP_FLAGS}" -o ../test_FULL_${SUFFIX}_hip_l0BE.csv "${HIP_TESTS[@]}"
fi

fi

#############################################

source /opt/intel/oneapi/setvars.sh

echo "@@@@@@@@@@@@@@@@@@@@@@@@@ SYCL DEVICES:"
sycl-ls
sleep 2


if [ $SYCL -ne 0 ]; then

# export SYCL_DUMP_IMAGES=1

# only for L0 backend
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

# enable SYCL cache in a random directory in TMP
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=/tmp/SYCL_CACHE_${RANDOM}

if [ $OPENCL -ne 0 ]; then
#  export SYCL_DEVICE_FILTER=opencl:gpu
  export ONEAPI_DEVICE_SELECTOR=opencl:gpu
  ./scripts/autohecbench.py --clean --warmup true --repeat ${REPEATS} --extra-compile-flags="${SYCL_FLAGS}" -o ../test_FULL_${SUFFIX}_sycl_oclBE.csv --sycl-type opencl sycl
fi

if [ $LEVEL0 -ne 0 ]; then
#  export SYCL_DEVICE_FILTER=ext_oneapi_level_zero:gpu:0
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
  ./scripts/autohecbench.py --clean --warmup true --repeat ${REPEATS} --extra-compile-flags="${SYCL_FLAGS}" -o ../test_FULL_${SUFFIX}_sycl_l0BE.csv --sycl-type opencl sycl
fi

fi

#############################################
