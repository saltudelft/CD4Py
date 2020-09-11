# License: GNU General Public License v3.0

from argparse import ArgumentParser
from cd4py.lib import deduplicate_py_data


def main():

    arg_parser = ArgumentParser(description="Code De-Duplication for Python")
    arg_parser.add_argument("--p", required=True, type=str, help="Path to Python projects")
    arg_parser.add_argument("--od", required=True, type=str, help="Output folder to store detected duplicate files.")
    arg_parser.add_argument("--ot", required=True, type=str, help="Output folder to store tokenized files.")
    arg_parser.add_argument("--d", default=2048, type=int, help="Dimension of TF-IDF vectors [default: 2048].")
    arg_parser.add_argument("--th", default=0.95, type=float,
                            help="Threshold to identify duplicate files [default: 0.95].")
    arg_parser.add_argument("--k", default=10, type=int, help="Number of nearest neighbor [default: 10].")
    arg_parser.add_argument("--tr", default=20, type=int,
                            help="Number trees to build the index. More trees gives higher precision but"
                                 " slower [default: 20].")

    args = arg_parser.parse_args()

    deduplicate_py_data(args.p, args.ot, args.od, args.d, args.th, args.k, args.tr)


if __name__ == '__main__':

    main()
