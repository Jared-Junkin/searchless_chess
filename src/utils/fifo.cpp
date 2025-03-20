#include <iostream>
#include <queue>
int main(){
    std::queue<int> q;
    for (int i = 0; i < 40; i+= 10){
        q.push(i); // insert
    }
    while (!q.empty()){
        std::cout << "First Element: " << q.front() << std::endl; // peek
        q.pop(); // Remove element
    }
    return 0;
}