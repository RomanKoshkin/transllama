#!/home/roman/miniconda3/envs/huggin/bin/python
# EASY-INSTALL-ENTRY-SCRIPT: 'simuleval','console_scripts','simuleval'
import re, argparse, sys, os, time
from termcolor import cprint

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", type=int, default=-1)
args, unknown_args = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)  # NOTE: this is set in the cli.py
cprint(os.getcwd(), color='magenta', attrs=['bold'])
time.sleep(5)

# for compatibility with easy_install; see #2198
__requires__ = 'simuleval'

try:
    from importlib.metadata import distribution
except ImportError:
    try:
        from importlib_metadata import distribution
    except ImportError:
        from pkg_resources import load_entry_point


def importlib_load_entry_point(spec, group, name):
    dist_name, _, _ = spec.partition('==')
    matches = (entry_point for entry_point in distribution(dist_name).entry_points
               if entry_point.group == group and entry_point.name == name)
    return next(matches).load()


globals().setdefault('load_entry_point', importlib_load_entry_point)

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    load_entry_point('simuleval', 'console_scripts', 'simuleval')()
    # sys.exit(load_entry_point('simuleval', 'console_scripts', 'simuleval')())
