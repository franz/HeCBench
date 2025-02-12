#!/usr/bin/env python3
#
# Script to run HeCBench benchmarks and gather results

import re, time, sys, subprocess, multiprocessing, os
import argparse
import json
import statistics

class Benchmark:
    def __init__(self, args, name, res_regex, run_args = [], binary = "main", invert = False):
        if name.endswith('sycl'):
            self.MAKE_ARGS = ['GCC_TOOLCHAIN="{}"'.format(args.gcc_toolchain)]
            if args.sycl_type == 'cuda':
                self.MAKE_ARGS.append('CUDA=yes')
                self.MAKE_ARGS.append('CUDA_ARCH=sm_{}'.format(args.nvidia_sm))
            elif args.sycl_type == 'hip':
                self.MAKE_ARGS.append('HIP=yes')
                self.MAKE_ARGS.append('HIP_ARCH={}'.format(args.amd_arch))
            elif args.sycl_type == 'opencl':
                self.MAKE_ARGS.append('CUDA=no')
                self.MAKE_ARGS.append('HIP=no')
                self.MAKE_ARGS.append('CC=icpx')
                self.MAKE_ARGS.append('CXX=icpx')
        elif name.endswith('cuda'):
            self.MAKE_ARGS = ['CUDA_ARCH=sm_{}'.format(args.nvidia_sm)]
        elif name.endswith('hip'):
            self.MAKE_ARGS = []
            self.MAKE_ARGS.append('CC=hipcc')
            self.MAKE_ARGS.append('CXX=hipcc')
        else:
            self.MAKE_ARGS = []

        if args.extra_compile_flags:
            flags = args.extra_compile_flags.replace(',',' ')
            self.MAKE_ARGS.append('EXTRA_CFLAGS={}'.format(flags))

        if name.endswith('opencl') and args.opencl_inc_dir:
            self.MAKE_ARGS.append('OPENCL_INC={}'.format(args.opencl_inc_dir))

        if args.bench_dir:
            self.path = os.path.realpath(os.path.join(args.bench_dir, name))
        else:
            self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', name)

        self.name = name
        self.binary = binary
        self.res_regex = res_regex
        self.args = run_args
        self.invert = invert
        self.clean = args.clean
        self.verbose = args.verbose

    def compile(self):
        if self.clean:
            subprocess.run(["make", "clean"], cwd=self.path).check_returncode()
            time.sleep(1) # required to make sure clean is done before building, despite run waiting on the invoked executable

        out = subprocess.DEVNULL
        if self.verbose:
            out = subprocess.PIPE

        proc = subprocess.run(["make"] + self.MAKE_ARGS, cwd=self.path, capture_output=True, encoding="ascii")
        try:
            proc.check_returncode()
        except subprocess.CalledProcessError as e:
            print(f'Failed compilation in {self.path}.\n{e}')
            if e.stdout:
                print(e.stdout, file=sys.stderr)
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            raise(e)

        if self.verbose:
            print(proc.stdout)

    def run(self, vtune_root_prefix = None, vtune_root_suffix = None,  numactl_args = None, extra_env = None):
        cmd = []
        if numactl_args:
            cmd.append("numactl")
            cmd.extend(numactl_args.split())
        if vtune_root_prefix:
            vtune_r = vtune_root_prefix + self.name
            if vtune_root_suffix:
                vtune_r += vtune_root_suffix
            cmd.extend(["vtune", '-collect', 'gpu-hotspots', '-r', vtune_r])
        cmd.append("./" + self.binary)
        cmd.extend(self.args)
        print("Running: " + " ".join(cmd))
        try:
            if extra_env is None:
                extra_env = os.environ.copy()
            proc = subprocess.run(cmd, cwd=self.path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                encoding="ascii", timeout=1200, env=extra_env)
            out = proc.stdout
            if proc.stderr:
                print("stderr output:", proc.stderr)
            if self.verbose:
                print("stdout output:", out)
            proc.check_returncode()
            res = re.findall(self.res_regex, out)
            if not res:
                raise Exception(self.path + ":\nno regex match for " + self.res_regex + " in\n" + out)
            res = sum([float(i) for i in res]) #in case of multiple outputs sum them
            if self.invert:
                res = 1/res
            return res
        except subprocess.TimeoutExpired as e:
            print(f"Timeout running {self.name}: {e}")
            if e.stdout:
                print("stdout:", e.stdout)
            if e.stderr:
                print("stderr:", e.stderr)
            raise
        except subprocess.CalledProcessError as e:
            print(f"Error running {self.name}. Return code: {e.returncode}")
            if e.stdout:
                print("stdout:", e.stdout)
            if e.stderr:
                print("stderr:", e.stderr)
            raise
        except Exception as e:
            print(f"Unexpected error running {self.name}: {e}")
            if hasattr(e, 'stdout') and e.stdout:
                print("stdout:", e.stdout)
            if hasattr(e, 'stderr') and e.stderr:
                print("stderr:", e.stderr)
            raise

    def dry_run(self):
        cmd = ["make", "run", "-n"]
        proc = subprocess.run(cmd, cwd=self.path, capture_output=True, encoding="ascii")
        try:
            proc.check_returncode()
        except subprocess.CalledProcessError as e:
            print(f'Failed dry run in {self.path}.\n{e}')
            if e.stdout:
                print(e.stdout, file=sys.stderr)
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            raise(e)
        
        # Extract the actual command being executed
        lines = proc.stdout.split('\n')
        for line in lines:
            if './' in line:  # This assumes the command starts with './'
                return line.strip()
        return None

def comp(b):
    print("compiling: {}".format(b.name))
    b.compile()

def should_skip(bench_name):
    """Check if a benchmark should be skipped based on known issues."""
    # Timeout prone benchmarks
    if (bench_name.startswith("tensorAccessor")
        # or bench_name.startswith("vanGenuchten") # chipStar fail with -cl-fast-relaxed-math
        # or bench_name.startswith("pnpoly") # chipStar fail with -cl-fast-relaxed-math
        # or bench_name.startswith("eigenvalue") # chipStar fail with -cl-fast-relaxed-math
        # or bench_name.startswith("entropy")): # chipStar fail with -cl-fast-relaxed-math
        or bench_name.startswith("lud") # verification takes forever
        or bench_name.startswith("matern")
        or bench_name.startswith("wyllie")
        or bench_name.startswith("rsbench")
        or bench_name.startswith("sort")):

        return "will likely timeout"

    # Double precision not supported
    if (bench_name.startswith("fresnel")
        or bench_name.startswith("goulash")
        or bench_name.startswith("burger")
        or bench_name.startswith("cooling")
        or bench_name.startswith("crossEntropy")
        or bench_name.startswith("fft")
        or bench_name.startswith("is")
        or bench_name.startswith("minisweep")
        or bench_name.startswith("pad")
        or bench_name.startswith("tsa")
        or bench_name.startswith("romberg")
        or bench_name.startswith("hellinger")
        or bench_name.startswith("tensorT")
        or bench_name.startswith("quicksort")):
        return "doubles not supported"

    # Hardware/environment requirements
    if bench_name.startswith("p2p"):
        return "requires multiple GPUs"

    # Known test failures
    if (bench_name.startswith("heat2d")      # fails in both SYCL and HIP
        or bench_name.startswith("aes")         # fails in SYCL, passes in HIP
        or bench_name.startswith("ced")         # fails in SYCL, passes in HIP
        or bench_name.startswith("f16max")      # passes in SYCL, fails in HIP
        or bench_name.startswith("shuffle")     # fails in SYCL, passes in HIP
        or bench_name.startswith("stencil1d")   # fails in SYCL, passes in HIP
        or bench_name.startswith("linearprobing") # fails in SYCL, passes in HIP
        or bench_name.startswith("snake")):       # passes in SYCL, fails in HIP
        return "test will fail"

    # Data issues
    if bench_name.startswith("bsw"):
        return "data file too large"
    if bench_name.startswith("columnarSolver") or bench_name.startswith("ans"):
        return "OOM"
    if bench_name.startswith("chi2"):
        return "can't find data"
    if bench_name.startswith("histogram"):
        return "OpenCL time overflow"

    return None

def main():
    parser = argparse.ArgumentParser(description='HeCBench runner')
    parser.add_argument('--output', '-o',
                        help='Output file for csv results')
    parser.add_argument('--repeat', '-r', type=int, default=1,
                        help='Repeat benchmark run')
    parser.add_argument('--warmup', '-w', type=bool, default=True,
                        help='Run a warmup iteration')
    parser.add_argument('--sycl-type', '-s', choices=['cuda', 'hip', 'opencl'], default='cuda',
                        help='Type of SYCL device to use')
    parser.add_argument('--nvidia-sm', type=int, default=60,
                        help='NVIDIA SM version')
    parser.add_argument('--amd-arch', default='gfx908',
                        help='AMD Architecture')
    parser.add_argument('--gcc-toolchain', default='',
                        help='GCC toolchain location')
    parser.add_argument('--opencl-inc-dir', default='/usr/include',
                        help='Include directory with CL/cl.h')
    parser.add_argument('--extra-compile-flags', '-e', default='',
                        help='Additional compilation flags (inserted before the predefined CFLAGS)')
    parser.add_argument('--clean', '-c', action='store_true',
                        help='Clean the builds')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Clean the builds')
    parser.add_argument('--bench-dir', '-b',
                        help='Benchmark directory')
    parser.add_argument('--bench-data', '-d',
                        help='Benchmark data')
    parser.add_argument('--bench-fails', '-f',
                        help='List of failing benchmarks to ignore')
    parser.add_argument('bench', nargs='+',
                        help='Either specific benchmark name or sycl, cuda, hip or opencl')
    parser.add_argument('--extra-env', default='',
                        help='Additional environment')
    parser.add_argument('--numactl-args', default=None,
                        help='numactl args')
    parser.add_argument('--vtune-root-prefix', default=None,
                        help='vtune report root directory base')
    parser.add_argument('--vtune-root-suffix', default=None,
                        help='vtune report root directory suffix ')
    parser.add_argument('--failing-only', action='store_true',
                        help='Run ONLY the tests that are known to fail or have issues')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load benchmark data
    if args.bench_data:
        bench_data = args.bench_data
    else:
        bench_data = os.path.join(script_dir, 'benchmarks', 'subset.json')

    with open(bench_data) as f:
        benchmarks = json.load(f)

    # Load fail file
    if args.bench_fails:
        bench_fails = os.path.abspath(args.bench_fails)
    else:
        bench_fails = os.path.join(script_dir, 'benchmarks', 'subset-fails.txt')

    with open(bench_fails) as f:
        fails = f.read().splitlines()

    # Build benchmark list
    benches = []
    for b in args.bench:
        if b in ['sycl', 'cuda', 'hip', 'opencl']:
            for k, v in benchmarks.items():
                if k.endswith(b) and k not in fails:
                    skip_reason = should_skip(k)
                    if args.failing_only:
                        if skip_reason:
                            print(f"Running known failing test {k}: {skip_reason}")
                            benches.append(Benchmark(args, k, *v))
                        continue
                    else:
                        if skip_reason:
                            print(f"Skipping {k}: {skip_reason}")
                            continue
                        benches.append(Benchmark(args, k, *v))
            continue

        skip_reason = should_skip(b)
        if args.failing_only:
            if skip_reason:
                print(f"Running known failing test {b}: {skip_reason}")
                benches.append(Benchmark(args, b, *benchmarks[b]))
            continue
        else:
            if skip_reason:
                print(f"Skipping {b}: {skip_reason}")
                continue
            benches.append(Benchmark(args, b, *benchmarks[b]))

    t0 = time.time()
    try:
        with multiprocessing.Pool(8) as p:
            p.map(comp, benches)
    except Exception as e:
        print("Compilation failed, exiting")
        print(e)
        sys.exit(1)

    t_compiled = time.time()
    if args.repeat == 0:
        print("compilation took {} s.".format(t_compiled-t0))
        print("Repeat value is zero. Exiting.")
        return

    outfile = sys.stdout
    existing = {}
    if args.output:
        if os.path.isfile(args.output):
            outfile = open(args.output, 'r+t')
        else:
            outfile = open(args.output, 'w+t')
        for line in outfile:
            bench, *rest = line.split(',')
            print("Found bench: {}", bench)
            existing[bench] = True
        outfile.seek(0, 2)

    extra_env = {}
    extra_env.update(os.environ)
    if args.extra_env:
        env_strs = args.extra_env.split(";")
        for e in env_strs:
            key, val = e.split("=", 1)
            extra_env[key] = val
    if args.numactl_args or args.vtune_root_prefix:
        args.warmup = False
        args.repeat = 1

    for i, b in enumerate(benches):
        try:
            print("\nrunning {}/{}: {}".format(i, len(benches), b.name), flush=True)
            if b.name in existing:
                print("result already exists, skipping", flush=True)
                continue
            time.sleep(1)

            if args.warmup:
                b.run()

            all_res = []
            for i in range(args.repeat):
                all_res.append(b.run(args.vtune_root_prefix, args.vtune_root_suffix,
                                     args.numactl_args, extra_env))
            # take the minimum result
            res_min = min(all_res)
            res_avg = sum(all_res) / len(all_res)
            if args.repeat > 1:
                res_stddev = statistics.stdev(all_res)
                res_coefvar = res_stddev / res_avg
            else:
                res_stddev = 0
                res_coefvar = 0

            print(b.name + "," + str(res_min)  + "," + str(res_avg)  + "," + str(res_stddev) +
                  "," + str(res_coefvar), file=outfile, flush=True)
        except Exception as err:
            print("Error running: {}".format(b.name), flush=True)
            print(err, flush=True)
            time.sleep(1)


    if args.output:
        outfile.close()

    t_done = time.time()
    print("compilation took {} s, runnning took {} s.".format(t_compiled-t0, t_done-t_compiled))



if __name__ == "__main__":
    main()
