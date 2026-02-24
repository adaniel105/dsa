#include <bits/stdc++.h>
using namespace std;

int main(){
    int n;
    int point_count = 0;
    cin >> n;
    vector<pair<int,int>>arr;
    for(int i = 0; i < n; i++){
        int a, b;
        cin >> a >> b;
        arr.push_back({a,b});
    }

    for(int i = 0; i < n; i++){
        int left = 0, right=0, up=0,down=0;
        int x = arr[i].first;
        int y = arr[i].second;
        for(int j = 0; j < n; j++){
            if(x<arr[j].first && y == arr[j].second){
                left++;
            }else if(x> arr[j].first && y == arr[j].second ){
                right++;
            }else if(x == arr[j].first && y < arr[j].second){
                up++;
            }else if(x == arr[j].first && y > arr[j].second){
                down++;
            }
            if(left>0 && right>0 && up>0 && down>0){
                if((left + right + up + down) >=4){
                    point_count++;      
                    break;          
                }
            }
        }
    }

    cout << point_count;
    return 0;
}
