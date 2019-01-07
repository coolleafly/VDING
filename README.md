# VDING
Precision Analysis Program for Dental Implant

## Prerequisites

- Python 3.4+ (Test in 3.6 )
- xlrd 1.1.0
- xlwt 1.3.0
- numpy 1.14.3+

## Usage

### Step1: fill in "implantData.xlsx" 

it is input file. it include settings and data for accuracy analysis.

Among them, it is easy to define:
- the number of implants
- the number of landmarks per implant
- the number of implant point of concern
- whether screw or abutment are included in the post surgery image


### Step2: Run program
    $ python main10.0.py


### Step3: get result
- out3* shows result of precision analysis and system error
- out1* shows points for establishing coordinate systems, and compute the coordinates of each point(Landmark or implant endpoints) in the new coordinate system
- out7* shows selection process of choose key point of establishing coordinate systems 

## More detail in our article, please wait it accepted and published.
