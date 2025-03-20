from typing import List, Tuple
import random
import matplotlib.pyplot as plt
import time

def boundedKnapsack(items: List[Tuple[int, int]], capacity)->int: # items must go (weight, value), so items[i][0] gives the weight of the ith item
    dp = [[0]*len(items) for _ in range(capacity+1)]
    for cap in range(capacity+1):
        for i in range(len(items)):

            if cap - items[i][0] >= 0:
                dp[cap][i] = max(dp[cap][i-1], dp[cap-items[i][0]][i-1]+items[i][1])
            else:
                dp[cap][i] = dp[cap][i-1]
    
    return dp[capacity][len(items)-1]

class Node:
    def __init__(self, key: int, value: int):
        self.key = key
        self.val = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node(0,0)
        self.tail = Node(0,0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add(self, node: Node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        # if key is already in our LRU cache, pop and then re-add (re-add part below)
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
        # of key is not in our LRU cache and we're at capacity, remove LRU node
        elif len(self.cache)>=self.capacity:
            lru_node = self.tail.prev
            self._remove(lru_node)
            del self.cache[lru_node.key]
        # add (or re-add) new node to LRU cache
        node = Node(key, value)
        self._add(node)
        self.cache[key] = node
        
        
def findElement(needle: int, haystack: List[int])->int:
    start, end = 0, len(haystack)-1
    while start < end:
        mid = (start+end)//2
        if haystack[mid]==needle:
            return mid
        elif haystack[mid]<needle:# we need to go up
            start = mid+1
        else:
            end = mid
    return -1
        # if haystack[start]<needle:
            
        
if __name__ == "__main__":
    ########## this will run binary search 
    # steps = 8
    # n = 10
    # runtimes = []
    # sizes = []

    # for i in range(steps):
    #     haystack = [0]  # Start with the first element
    #     for _ in range(1, n):
    #         haystack.append(haystack[-1] + random.randint(1, 10))  # Increment by a random value
    #     sizes.append(n)
    #     start = time.time()
    #     res = findElement(haystack[3], haystack)
    #     end = time.time()
    #     runtimes.append(end - start)
    #     print(f"searching {n} elements took {end-start} seconds")
    #     n *= 10

    # # Plotting the results
    # plt.figure()
    # plt.plot(sizes, runtimes, marker="o")
    # plt.xlabel("Size of haystack (n)")
    # plt.ylabel("Runtime (seconds)")
    # plt.title("Runtime Analysis of findElement")
    # plt.xscale("log")  # Log scale for n
    # plt.grid(True)
    # plt.savefig("./runtime.png")
    # print("Runtime plot saved as './runtime.png'")
    ########## bounded knapsack problem
    items = [(7,7), (3,3), (5,5)]
    capacity = 8
    print(boundedKnapsack(items, capacity))