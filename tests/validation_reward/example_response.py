def max_buildings_with_illumination():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    heights = list(map(int, data[1:]))
    
    max_count = 1
    
    for start in range(N):
        for interval in range(1, N):
            count = 1
            current_height = heights[start]
            next_index = start + interval
            
            while next_index < N:
                if heights[next_index] == current_height:
                    count += 1
                    next_index += interval
                else:
                    break
            
            max_count = max(max_count, count)
    
    print(max_count)

if __name__ == '__main__':
    max_buildings_with_illumination()