if __name__ == '__main__':
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Find the largest element
    max_element = max(A)
    
    # Remove the largest element and find the second largest
    A.remove(max_element)
    second_largest_element = max(A)
    
    # Find the index of the second largest element in the original list
    # We add 1 because the problem expects 1-based index
    second_largest_index = A.index(second_largest_element) + 1
    
    print(second_largest_index)