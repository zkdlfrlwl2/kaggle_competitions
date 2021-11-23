#### Chris Deotte Tips list-up

* https://www.kaggle.com/cdeotte
* Chris Deotte의 notebook & discussion 정리한 내용
* When using NN with regression RSME, we may need to set dropout=0.





#### What may be the reasons  for the CV-LB gap?

* https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/284172
* Maybe the LB dataset(hope not the final dataset) has a really different distribution with the training set.
* CV 계산법이 잘못됐을 가능성 존재
  * RMSE의 경우, batch 별 계산 대신 fold 별 계산을 해야하고 전체 CV 계산도 fold 별 값을 평균내는 것이 아니고 전 oof를 대상으로 해야한다.
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/289790




### Colab 연결 끊김 방지

```javascript
function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
}
setInterval(ConnectButton,60000);
```

