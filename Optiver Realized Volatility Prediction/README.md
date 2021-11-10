##  Optiver Realized Volatility Prediction
* https://www.kaggle.com/c/optiver-realized-volatility-prediction

### Time line

* Training Timeline
  * June 28, 2021 - Start Date
  * September 27, 2021 - Final submission deadline
* Forecasting Timeline
  * January 10, 2022 - Competition end date



### Model

* LGBM
* FCNN with skip connection
* Ensemble of above models



### Model Architecture



![overall](https://user-images.githubusercontent.com/92927837/141056292-e0190a00-9d4a-4222-a346-6ed7006c8b99.png)



![base](https://user-images.githubusercontent.com/92927837/141055960-0f14f301-e81b-4963-962f-122e3caf7af9.png)

* 위 모델 구조를 베이스로 Custom Layer만 변경하여 성능 테스트하였음

  

![cl1](https://user-images.githubusercontent.com/92927837/141056046-5f6e6cc6-f552-49b5-bdff-4a1a499cd964.png)

* Custom Layer 1의 경우, 가장 기본이 되는 Dense Layer 구조이며 최적화를 통해 CV/LB=0.20879/0.19966으로 LB 점수 0.2x ↓ 도달하였음

  

  ![cl2](https://user-images.githubusercontent.com/92927837/141056097-0e0fa353-d399-412c-91b5-de055b715702.png)

* Custom Layer 2의 경우, 가우시안 노이즈를 추가하여 과적합을 완화하려 했으나 오히려 Custom Layer 1보다 성능이 낮게 나왔음 

  

  ![cl3](https://user-images.githubusercontent.com/92927837/141056146-9851b77d-af6b-49d2-a5a5-cf7a5e098868.png)

* Custom Layer 3의 경우, 골자가 되는 Custom Layer 1을 보강한 형태로 결과는 CV/LB=0.20884/0.19836로 세 Custom Layer 중 가장 성능이 높게 나왔음



### Result

* Training Phase - End

  ![image](https://user-images.githubusercontent.com/92927837/141051093-62c0fbbb-b275-4014-b403-a92200185e0e.png)

* Forecasting Phase - Ongoing

  ![image](https://user-images.githubusercontent.com/92927837/141051181-742db049-a685-4262-9e38-2b73d2ceff5e.png)





### Overview

* 주식 기초 상식
  
  * bid (입찰): **매수자가 주식에 기꺼이 지불하고자 하는 최대 금액**
  * ask (요청): **주식 소유자가 판매하고자 하는 최소 금액**
  * ask 가격은 항상 bid 가격보다 약간 높다
  * spread: ask가 10달러, bid가 10.05달러면 0.05달러가 spread, spread가 크면 유동성이 적고 spread가 작으면 유동성이 크다. 거래가 활발하면 ask와 bid 사이의 가격 차이가 적다
  
* **purpose**: In this competition, you will be given 10 minutes of book data and we ask you to predict what the volatility will be in the following 10 minutes. / In this competition, Kagglers are challenged to generate a series of short-term signals from the book and trade data of a fixed 10-minute window to predict the realized volatility of the next 10-minute window.
  
* **target**: *Realized volatility of the next 10 minute window* under the same stock_id/time_id
  
* **loss**: RMSPE - root mean square percentage error
  
* **Data Description**
  
  * stock_id: is a unique identifier of a stock in real life, and the group of 112 unique stock_id will be present in all datasets.
  
  * time_id: **represent one unique time window in real life** and was shuffled randomly so they are not sequential at all,  represents a **unique 20-minutes trading window** which is consistent across all stocks, The data in the **first 10 minutes window is shared with all of you**, while the order book data of the **second 10-minutes is used to build the target for you to predict**.
  
    
  
    ![TimeID](https://user-images.githubusercontent.com/92927837/141054778-f3646b1d-d3ea-40ae-a8c3-43eef4082665.png)
  
   
  
  * seconds_in_bucket: Number of seconds from the start of the bucket
  * bid_price[1/2]: Normalized prices of the most/second most competitive buy level
  * ask_price[1/2]: Normalized prices of the most/second most competitive sell level
  * bid_size[1/2]: The number of shares on the most/second most competitive buy level
  * ask_size[1/2]: The number of shares on the most/second most competitive sell level
  * price: The average price of executed transactions happening in one second, Prices have been normalized and the average has been weighted by the number of shares traded in each transaction, the price is aggregated as a **volume weighted average price** of all trades
  * size: The sum number of shares traded
  * order_count: The number of unique trade orders taking place
  
* **Order book**

  * refers to an electronic list of buy and sell orders for a specific security (특정 증권) or financial instrument (금융 상품) organized by price level. An order book lists the number of shares (주식) being bid on (입찰) or offered (제공) at each price point.

  * Below is a snapshot of an order book of a stock (let's call it stock A), as you can see, all intended buy orders are on the left side of the book displayed as "bid" while all intended sell orders are on the right side of the book displayed as "offer/ask"

     

  ![OrderBook3](https://user-images.githubusercontent.com/92927837/141054922-f8d19abb-86e4-4f67-b008-9995c5cf866f.png)

  

  * An actively traded financial instrument (금융 상품) always has a dense order book (A liquid book). As the order book data is a continuous representation of market demand/supply it is always considered as the number one data source for market research.

* **Trade**

  * An order book is a representation of trading intention (의도) on the market, however the market *needs a buyer and seller at the **same price** to make the trade happen*. Therefore, sometimes when someone wants to do a trade in a stock, they check the order book and find someone with counter-interest to trade with.

  * For example, imagine you want to buy 20 shares (주식) of a stock A when you have the order book in the previous paragraph. Then you need to find some people who are willing to trade against you by selling 20 shares or more in total. You check the **offer** side of the book starting from the lowest price: there are 221 shares of selling interest on the level of 148. You can **lift** 20 shares for a price of 148 and **guarantee** your execution. This will be the resulting order book of stock A after your trade:

     

     ![OrderBook4](https://user-images.githubusercontent.com/92927837/141054999-6db41417-cccb-4ee2-a475-9673aff4e4aa.png)

     

  * Similar to order book data, trade data is also extremely crucial to Optiver's data scientists, as it reflects how active the market is. Actually, some commonly seen technical signals of the financial market are derived from trade data directly, such as high-low or total traded volume.

* **Market making and market efficiency**

  * Imagine, on another day, stock A's order book becomes below shape, and you, again, want to buy 20 shares from all the intentional sellers. As you can see the book is not as dense as the previous one, and one can say, compared with the previous one, this book is **less liquid**.

    

    ![OrderBook5](https://user-images.githubusercontent.com/92927837/141055326-187ebbde-eeba-444f-8f01-c8768b3adf9a.png)

    

  * You could insert an order to buy at 148. However, there is nobody currently willing to sell to you at 148, so your order will be sitting in the book, waiting for someone to trade against it. If you get unlucky, the price goes up, and others start bidding at 149, and you never get to buy at all. (매수하지 못한다) Alternatively, you could insert an order to buy at 155. The exchange would match this order against the outstanding sell order of one share at 149, so you buy 1 lot at 149. Similarly, you'd buy 12 shares at a price of 150, and 7 shares at 151. Compared to trying to buy at 148, there is no risk of not getting the trade that you wanted, but you do end up buying at a higher price. (매수는 할 수 있으나 비싼 가격에 매수하게 된다)

* **Order book statistics**

  * There are a lot of statistics Optiver data scientist can derive from raw order book data to reflect market liquidity and stock valuation.

  * **bid/ask spread**

    * As different stocks trade on different level on the market we take the ratio of best offer price and best bid price to calculate the bid-ask spread.

      ![bidask](https://user-images.githubusercontent.com/92927837/141055437-e2ec6ef0-0546-4dd5-8719-00ae7d6f9a63.png)

  * **Weighted averaged price - WAP**

    * to calculate the instantaneous stock valuation and calculate realized volatility as our target.

      ![wap](https://user-images.githubusercontent.com/92927837/141055632-a01564b5-e107-45b0-b8ff-696b5aa6652d.png)

  * **Log returns**

    * *How can we compare the price of a stock between yesterday and today ?*

      ![logreturn](https://user-images.githubusercontent.com/92927837/141055635-69ab2413-1075-4814-af63-b377bb6fb019.png)

  * **Realized volatility**

    * *volatility* is the annualized standard deviation of stock log returns - normalized to a 1 year period because the standard deviation will be different for log returns computed over longer or shorter intervals.

      ![rv](https://user-images.githubusercontent.com/92927837/141055638-1a45f05b-f55e-47fc-8073-30933c4c55a2.png)

* Reference

  * https://www.kaggle.com/jiashenliu/introduction-to-financial-concepts-and-data

