#include <iostream>
#include <vector>

// def isInList(num: int, L: List)->bool:
//   for i in L:
//     if num == i:
//       return True
//   return False

bool isInList(int num, int* L, int length){
    int start = 0;
    int end = length;
    while (start <= end){
        int mid = (start + end)/2;
        if (L[mid] == num){
            return true;
        }
        else if (L[mid] < num){
            start = mid+1;
        }
        else if (L[mid]>num){
            end = mid-1;
        }
    }
    return false;

    // for (int i=0; i<length; i++){
    //     if (L[i] == num){
    //         return true;
    //     }
    // }

    // for (const auto& n : L){
    //     if (n==num){
    //         return true;
    //     }
    // }
    return false;
}

int main(){


    int num = 2;
    int* L = (int*)malloc(3*sizeof(int));
    for (int i=0; i<3; i++){
        L[i]=i+1;
    }

    std::cout << isInList(num, L, 3) << std::endl;
    num=4;
    std::cout << isInList(num, L, 3) << std::endl;
    std::cout << &L << std::endl;
    free(L);
    // int num2=4;
    // std::cout << isInList(num2, L) << std::endl;
}