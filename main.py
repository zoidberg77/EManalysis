import os, sys
import argparse

from analyzer.data import Dataloader

def get_args():
    '''
    Get arguments from command lines.
    '''
    parser = argparse.ArgumentParser(description="Model for clustering mitochondria.")
    parser.add_argument('--input', type=str, help='input directory (path)')

def main():
    '''
        Main function.
    '''
    # input arguments are parsed.
    args = get_args()
    print("Command line arguments:")
    print(args)

    dl = Dataloader()



if __name__ == "__main__":
    main()
