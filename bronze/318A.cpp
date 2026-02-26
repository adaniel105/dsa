//simulated the entire thing instead of actually doing the math, replace later
#include <bits/stdc++.h>
using namespace std;

int main(){
    long long n , k;
    cin >> n >> k;

        if(n % 2 == 0){
            if(k > (n/2)){
                cout << 2 * (k - (n/2));
                return 0;
            }else{
                if(k == 1){
                    cout << k;
                    return 0;
                }else if(k == 2){
                    cout << k + 1;
                    return 0;
                }
                cout << k + (k - 1);
                return 0;
            }
        }else{
            if( k > (n/2) + 1){
                cout << 2 * (k - ((n/2) + 1));
                return 0;
            }else{
                if(k == 1){
                    cout << k;
                    return 0;
                }else if(k == 2){
                    cout << k + 1;
                    return 0;
                }
                cout << k + (k - 1);
                return 0;
            }
        }

}
