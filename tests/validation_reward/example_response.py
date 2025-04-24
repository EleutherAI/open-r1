if __name__ == '__main__':
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Initialize variables to track the largest and second largest values
    largest = second_largest = float('-inf')
    
    # Iterate through the list to find the largest and second largest elements
    for num in A:
        if num > largest:
            second_largest = largest
            largest = num
        elif num > second_largest:
            second_largest = num
    
    # Find the index of the second largest element (1-based index)
    index_of_second_largest = A.index(second_largest) + 1
    
    # Print the result
    print(index_of_second_largest)