# Industry Cooperation - Credit_rating

## Use base_processor.py
- 使用 base_processor.py中的抽象類別 FeaturesProcessor 去延伸實作特徵處理，FeaturesProcessor要實作，對特徵資料的處理，
  對Y資料的處理(quota)，對職稱類別資料的分類方式，只要這三個有實作，剩下的可以請使用者自行增加其他功能，若有需要覺得必須要實作的也歡迎提出討論
- 另一個抽象類別是PredictModel，實作出怎麼讀取model，並將處理好的資料丟到Model中，返回預測的結果。預測的結果必須包含身分證、預測的額度。

## category.json
- 此資料是代碼類別定義，裡面有每個只要是代碼值所對應的是什麼，若有缺漏可以再跟我說