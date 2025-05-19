# [Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview)

by: [ryantjx](https://github.com/ryantjx)

#### Important Links
- [Optiver - Trading At The Close Introduction](https://www.kaggle.com/code/tomforbes/optiver-trading-at-the-close-introduction)
  - Auction
    - In a closing auction, orders are collected over a pre-determined timeframe and then matched at a single price determined by the buy & sell demand expressed by auction participants.
    - The closing price is determined as: The price at which the maximum number of shares can be matched.
  - CLOB - Bid does not cross Ask. Vice Versa.
  - Auction Order Book
    - In this book, the orders are not immediately matched, but instead collected until the moment the auction ends. The book in the example below is referred to as **in cross**, since the best bid and ask are overlapping. 
    - The closing auction price is therefore referred to as the **uncross** price, the price at which the shares which were in cross are matched.
    - **matched size** - quantity successfully matched between buy and sell orders.
    - The term **imbalance** refers to the number of unmatched shares. At the uncross price of 8, there are 7 bids & 4 asks which can be matched, therefore we are left with 3 bids unmatched. Since bids are orders to buy, there is an imbalance of 3 lots in the buy direction.
    - The term **far price** refers to the hypothetical uncross price of the auction book, if it were to uncross at the reporting time. Nasdaq provides far price information 5 minutes before the closing cross.
  - Combined Book
    - The hypothetical uncross price of combined book is called the near price. Nasdaq provides near price information 5 minutes before the closing cross.
    - Nasdaq also provides an indication of the fair price called the reference price. The reference price is calculated as follows:
      - If the near price is between the best bid and ask, then the reference price is equal to the near price
      - If the near price > best ask, then reference price = best ask
      - If the near price < best bid, then reference price = best bid So the reference price is the near price bounded between the best bid and ask.
    - Key Information
      - However, we have some information in our dataset that should help us to beat this baseline. If we observe an auction imbalance, it indicates that at the current price there is buying or selling interest that will currently not get matched in the auction. We can therefore adjust our prediction upwards if there is a buy imbalance & downwards if there is a sell imbalance.
- [Winning Solution - hyd](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446)
  - My final model(CV/Private LB of 5.8117/5.4030) was a combination of CatBoost (5.8240/5.4165), GRU (5.8481/5.4259), and Transformer (5.8619/5.4296), with respective weights of 0.5, 0.3, 0.2 searched from validation set. And these models share same 300 features.
  - torch.transformerencoder
  - Retrain model every 12 days, 5 times in total.
  - What did not work
    - ensemble with 1dCNN or MLP.
    - multi-days input instead of singe day input when applying GRU models
    - larger transformer, e.g. deberta 
    - predict target bucket mean by GBDT(gradient boosting decision tree)
- [9th-Place Solution - ADAM.](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486868)
  - Xgboost with 3 different seeds and same 157 features

#### Extra Links
- [The Informational Content of an Open Limit Order Book](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=565324)
  - (1) Does the limit order book allow better inferences about a security's value than simply the best bid and offer prices from the first step of the book? If it does, how much additional information can be gleaned from the book? 
  - (2) Are imbalances between the demand and supply schedules informative about future price movements? and 
  - (3) Does the shape of the limit order book impact traders' order submission strategies? 
  - Our empirical evidence suggests that the order book beyond the first step is informative - its information share is about 30%.
