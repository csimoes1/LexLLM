#!/usr/bin/env python3
import sys

def find_longest_line(filename):
    longest_line = ""
    with open(filename, 'r') as file:
        for line in file:
            # Remove the newline character for accurate length comparison
            current_line = line.rstrip('\n')
            if len(current_line) > len(longest_line):
                longest_line = current_line
    return longest_line

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        longest = find_longest_line(filename)
        print(f"Longest line:{len(longest)}")
        # print(longest)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
