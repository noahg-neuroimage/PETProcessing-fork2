import argparse
import json
from . import preproc

_PREPROC_EXAMPLES_ = (r"""
Examples:
  - Create a blank params file:
    pet-cli-preproc -p path_to_new_params.json
  - Run one or more preprocessing methods in order:
    pet-cli-preproc -p path_to_existing_params.json -m [method1] [method2] ...
""")


def create_blank_properties(properties_file: str):
    preproc_props = preproc.PreProc._init_preproc_props()
    with open(properties_file,'w') as f:
        json.dump(preproc_props,f,indent=4)


def main():
    parser = argparse.ArgumentParser(prog='pet-cli-preproc',
                                     description='Preprocessing command line interface',
                                     epilog=_PREPROC_EXAMPLES_,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p','--properties',help='Path to a properties file')
    parser.add_argument('-m','--method_name',help='Name of process or method to run')
    args = parser.parse_args()

    with open(args.properties,'r') as f:
        preproc_props = json.load(f)


    subject = preproc.PreProc(output_directory='',output_filename_prefix='')
    subject.update_props(new_preproc_props=preproc_props)
    subject.run_preproc(method_name=args.method_name)
