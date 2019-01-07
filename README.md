# VDING
Precision Analysis Program for Dental Implants

## Prerequisites

- Python 3.4+ (Test in 3.6 )
- xlrd 1.1.0
- xlwt 1.3.0
- numpy 1.14.3+

## Usage

### Step1: fill in the "implantData.xlsx" 

It is an input file. It include settings and data for accuracy analysis.

Among them, it is easy to define:
- the number of implants
- the number of landmarks per implant
- the number of points of concern on the implant
- whether screw or abutment are included in the postoperative image


### Step2: Run program
    $ python main10.0.py


### Step3: get result
- out3* shows result of precision analysis and system error
- out1* shows the points for establishing coordinate systems, and the coordinates of each point(Landmarks or implant endpoints) in the new coordinate system
- out7* shows the selection process of key points for establishing the coordinate systems 

## More detail in our article, please wait it accepted and published.
(Ye Liang, Shanshan Yuan and etc.)
