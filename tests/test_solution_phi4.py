from collections import defaultdict
from itertools import combinations

def solve_web_group_identification():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    n = int(data[0])
    addresses = data[1:]
    
    # Dictionary to store paths for each hostname
    hostname_to_paths = defaultdict(set)
    
    # Parse each address
    for address in addresses:
        if address.startswith("http://"):
            # Remove the protocol part
            address = address[7:]
            
            # Split hostname and path
            if '/' in address:
                hostname, path = address.split('/', 1)
            else:
                hostname, path = address, ''
            
            # Add the path to the corresponding hostname's set
            hostname_to_paths[hostname].add(path)
    
    # Dictionary to map paths sets to hostnames
    paths_to_hostnames = defaultdict(set)
    
    # Group hostnames by their paths set
    for hostname, paths in hostname_to_paths.items():
        paths_to_hostnames[frozenset(paths)].add(hostname)
    
    # Collect groups of hostnames that share the same paths set
    groups = [hostnames for hostnames in paths_to_hostnames.values() if len(hostnames) > 1]
    
    # Output the result
    print(len(groups))
    for group in groups:
        print(" ".join(f"http://{hostname}" for hostname in sorted(group)))

if __name__ == '__main__':
    solve_web_group_identification()
