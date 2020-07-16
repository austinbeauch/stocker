# STOCKER

A very bad stock prediction model. Goals of the project include

* Making a trillion dollars
* Implementing prediction and trading models, using recurrent neural networks and reinforcement learning, respectively
* Building web scrapers (possibly passing headlines through a CNN) to generate more input data

### V0.1
* The worst
* Input n-day moving average, output of the next k-days moving average prices
* Tries to predict absolute values (emphasis on tries)
* Generally terrible all around

### Planned Future Versions
* Aim to predict single day price *changes* instead of absolute values
* Can call model repeatedly, inputting the last predicted price, to generate a sequence of future prices
* Associate a confidence metric that can compound over time, resulting in longer predictions being less likely 
* Add in supplemental information - news headlines, Reddit comments, etc
    * Either hand-selected features (polarity), or pass through another network for processing