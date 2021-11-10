##  Google Brain - Ventilator Pressure Prediction

### Time line

* **September 22, 2021** - Start Date.
* **October 27, 2021** - Entry Deadline. You must accept the competition rules before this date in order to compete.
* **October 27, 2021** - Team Merger Deadline. This is the last day participants may join or merge teams.
* **November 3, 2021** - Final Submission Deadline.



###  Evaluation

* Mean absolute error, only the inspiratory(흡기) phase of each breath not expiratory(배기) phase.



### Description

* developing new methods for controlling mechanical ventilators is prohibitively expensive, even before reaching clinical trials. High-quality simulators could reduce this barrier.
* believe that neural networks and deep learning can better generalize across lungs with varying characteristics than the current industry standard of PID controllers.
* In this competition, you’ll simulate a ventilator connected to a sedated patient's lung. The best submissions will take lung attributes compliance and resistance into account.



### Units overview

* H2O
  * chemical formula for water

* cmH2O
  * a unit to measure **pressure**
  * 1cmH2O
    * the pressure exerted by a column of water of height 1cm
    * 높이 1cm 물기둥이 가하는 압력
* L
  * liters
  * 1L = 1,000mL
* S
  * seconds
  * measure of time
* cmH2O/L/S
  * **airway resistance**
    * 기도 저항
    * pressure (cmH2O) over flux-유량(liter/second)
    * 유량에 의한 압력
  * cmH20/L/S will give you the change in pressure per change in airflow
    * 기류 변화에 따른 압력 변화
  * (cmH2O) / (L/S)
  * L/S
    * volume of air flowing through the lungs per seconds
    * measure airflow
* mL/cmH2O
  * **compliance**
    * 순응도?
    * ratio between changes of volume(mL) and pressure(cmH2O)
      * 부피와 압력의 변화 비율
  * mL/cmH20 will give you the change in volume (mL) per change in pressure (cmH20)
    * 압력 변화(cmH2O)당 부피 변화(mL)



### Data overview

* `id` - globally-unique time step identifier across an entire file
  * row 갯수
* `breath_id` - globally-unique time step for breaths
  * 3초마다 갱신되는 호흡 횟수
* `R` - lung attribute indicating how restricted the airway is (in **cmH2O/L/S**). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with **higher `R` being harder to blow.**
  * **airway resistance**
* `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with **higher `C` having thinner latex and easier to blow.**
  * **compliance**
* `time_step` - the actual time stamp.
  * 측정되는 순간의 time stamp
* `u_in` - the control input for the inspiratory solenoid valve(흡기 밸브). Ranges from 0 to 100.
  * The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., **0 is completely closed and no air is let in and 100 is completely open**).
* `u_out` - the control input for the exploratory solenoid valve(배기 밸브). Either 0 or 1.
  * The second control input is a binary variable representing whether **the exploratory valve is open (1) or closed (0) to let air out.**
* `pressure` - the **airway pressure**(기도 압력) measured in the respiratory circuit, measured in **cmH2O**(단위).
* ![image-20210927142147039](C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20210927142147039.png)
* ![image-20210927142929038](C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20210927142929038.png)



### Result

![result](C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20211110111241668.png)

