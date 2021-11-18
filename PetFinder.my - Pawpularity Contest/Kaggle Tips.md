#### Chris Deotte Tips list-up

* https://www.kaggle.com/cdeotte
* Chris Deotte의 notebook & discussion 정리한 내용
* When using NN with regression RSME, we may need to set dropout=0.





#### What may be the reasons  for the CV-LB gap?

* https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/284172
* Maybe the LB dataset(hope not the final dataset) has a really different distribution with the training set.



### Colab 연결 끊김 방지

```javascript
function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
}
setInterval(ConnectButton,60000);
```

