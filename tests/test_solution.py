import sys
from collections import defaultdict

def solve():
    n = int(sys.stdin.readline())

    host_paths = defaultdict(set)

    for _ in range(n):
        url = sys.stdin.readline().strip()

        # Remove "http://" prefix
        if url.startswith("http://"):
            url = url[7:]
        else:
            # Handle cases where the prefix might be missing or different, though problem statement implies http://
            # For robustness, find the first occurrence of :// and take the part after it.
            # If not found, assume the whole string is the host part (less likely based on problem desc)
            proto_end = url.find("://")
            if proto_end != -1:
                url = url[proto_end + 3:]
            # else: url remains as is, assuming it's just hostname[/path]

        slash_index = url.find('/')

        if slash_index == -1:
            hostname = url
            # Use a distinct marker for requests like http://hostname (no trailing slash)
            path = "_ROOT_"
        else:
            hostname = url[:slash_index]
            path = url[slash_index:] # Keep the leading slash, e.g., "/", "/test"

        host_paths[hostname].add(path)

    # Group hostnames by their path sets
    pathset_hosts = defaultdict(list)
    for hostname, paths in host_paths.items():
        # Use frozenset as dict key requires hashable type
        frozen_paths = frozenset(paths)
        pathset_hosts[frozen_paths].append(hostname)

    # Filter groups with size > 1
    groups = []
    for hosts in pathset_hosts.values():
        if len(hosts) > 1:
            groups.append(hosts)

    # Print the output
    print(len(groups))
    groups = sorted(groups, key=lambda x: x[0])
    for group in groups:
        # Prepend "http://" back as shown in the example output
        print(" ".join(f"http://{host}" for host in group))

# if __name__ == "__main__":
#     solve()