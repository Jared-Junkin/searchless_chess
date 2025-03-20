def iterate(nums):
    slow = nums[0]  # Start slow at first index
    fast = nums[0]  # Start fast at first index

    # Step 1: Move slow and fast at least once before checking
    slow = nums[slow]
    fast = nums[nums[fast]]

    i = 1  # We've already taken one step
    while slow != fast:
        slow = nums[slow]          # Move slow one step
        fast = nums[nums[fast]]    # Move fast two steps
        if i % 1000 == 0:
            print(f"{i} iterations have passed")
        i += 1

    print(f"Breaking out at {nums[slow]} after {i} steps")
    print(f"nums[i] is {slow}, numf[fast] is {fast}.")
        
if __name__ == "__main__":
    # nums = [3,1,2,3,4]
    # iterate(nums)
    hashmap = {'a':5, 'b':3, 'c':-10}
    res = sorted(hashmap, key = hashmap.get)
    print(res)