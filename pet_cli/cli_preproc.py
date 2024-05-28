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


def create_blank_params(params_file: str):
    preproc_props = preproc._PREPROC_PROPS_
    with open(params_file,'w') as f:
        json.dump(preproc_props,f,indent=4)


def main():
    parser = argparse.ArgumentParser(prog='pet-cli-preproc',
                                     description='Preprocessing command line interface',
                                     epilog=_PREPROC_EXAMPLES_,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p','--params',help='Path to a params file',required=True)
    parser.add_argument('-m','--method',help='Name of process or method to run',nargs='*',required=False)
    args = parser.parse_args()

    if args.method is None:
        create_blank_params(args.params)
        return 0

    with open(args.params,'r') as f:
        preproc_props = json.load(f)

    subject = preproc.PreProc(output_directory='',output_filename_prefix='')
    subject.update_props(new_preproc_props=preproc_props)



    methods = args.method
    for method in methods:
        subject.run_preproc(method_name=method)


if __name__ == "__main__":
    main()