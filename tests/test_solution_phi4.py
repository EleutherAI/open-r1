# from collections import defaultdict
# from itertools import combinations

# def solve_web_group_identification():
#     import sys
#     input = sys.stdin.read
#     data = input().splitlines()
    
#     n = int(data[0])
#     addresses = data[1:]
    
#     # Dictionary to store paths for each hostname
#     hostname_to_paths = defaultdict(set)
    
#     # Parse each address
#     for address in addresses:
#         if address.startswith("http://"):
#             # Remove the protocol part
#             address = address[7:]
            
#             # Split hostname and path
#             if '/' in address:
#                 hostname, path = address.split('/', 1)
#             else:
#                 hostname, path = address, ''
            
#             # Add the path to the corresponding hostname's set
#             hostname_to_paths[hostname].add(path)
    
#     # Dictionary to map paths sets to hostnames
#     paths_to_hostnames = defaultdict(set)
    
#     # Group hostnames by their paths set
#     for hostname, paths in hostname_to_paths.items():
#         paths_to_hostnames[frozenset(paths)].add(hostname)
    
#     # Collect groups of hostnames that share the same paths set
#     groups = [hostnames for hostnames in paths_to_hostnames.values() if len(hostnames) > 1]
    
#     # Output the result
#     print(len(groups))
#     for group in groups:
#         print(" ".join(f"http://{hostname}" for hostname in sorted(group)))

# if __name__ == '__main__':
#     solve_web_group_identification()

import math

def min_turns_for_chess_coloring(T, grid_sizes):
    results = []
    for n in grid_sizes:
        # Calculate the minimum number of turns
        turns = math.floor(n / 2) + 1
        results.append(turns)
    return results

if __name__ == '__main__':
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    T = int(data[0])
    grid_sizes = [int(data[i]) for i in range(1, T + 1)]
    results = min_turns_for_chess_coloring(T, grid_sizes)
    for result in results:
        print(result)


