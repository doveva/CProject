import urllib3
import requests
import datetime


#filename = str(d) + "." + str(m) + "." + str(y) + ".csv"
#folder = "..\Anaconda3\Data"
#while y<2020:
#    while m<=12:
#        while d<=31:
#            if m <10 and d<10:
#                url = "http://so-ups.ru/index.php?id=oes_east_gen_consump_hour&tx_ms1cdu_pi1[dt]=" + "0" + str(d) + "." + "0" +str(m) + "." + str(y) + "&tx_ms1cdu_pi1[format]=csv"
#            elif m<10 and d>=10:
#                url = "http://so-ups.ru/index.php?id=oes_east_gen_consump_hour&tx_ms1cdu_pi1[dt]=" + str(d) + "." + "0" + str(m) + "." + str(y) + "&tx_ms1cdu_pi1[format]=csv"
#            elif m >= 10 and d < 10:
#                url = "http://so-ups.ru/index.php?id=oes_east_gen_consump_hour&tx_ms1cdu_pi1[dt]=" + "0" + str(d) + "." +  str(m) + "." + str(y) + "&tx_ms1cdu_pi1[format]=csv"
#            else:
#                url = "http://so-ups.ru/index.php?id=oes_east_gen_consump_hour&tx_ms1cdu_pi1[dt]=" + str(d) + "." + str(m) + "." + str(y) + "&tx_ms1cdu_pi1[format]=csv"
#            filename = str(d) + "." + str(m) + "." + str(y) + ".csv"
#            r = requests.get(url, verify=False)
#            with open(filename, 'wb') as f:
#                f.write(r.content)
#            print("Now connected to " + url + " | writing to " + filename)
#            d = d + 1
#        m=m+1
#        d=1
#    y=y+1
#    m=1

http = urllib3.PoolManager()
d: int
m: int
y: int
d1: int
m1: int
y1: int

d = 5
m = 4
y = 2001
d2 = 31
m2 = 12
y2 = 2019


y1=datetime.date(y,m,d)
y2=datetime.date(y2,m2,d2)

diff=datetime.timedelta(1)

y3=y2-y1
d=str(y3)
d1=int(d[0]+d[1]+d[2]+d[3])+2
y_n=y1

folder = "..\Data"

for i in range(1,d1):
    if y_n.month<10:
        url="http://so-ups.ru/index.php?id=oes_east_gen_consump_hour&tx_ms1cdu_pi1[dt]=" + str(y_n.day)+ ".0" + str(y_n.month) + "." + str(y_n.year) + "&tx_ms1cdu_pi1[format]=csv"
        if y_n.day<10:
            url = "http://so-ups.ru/index.php?id=oes_east_gen_consump_hour&tx_ms1cdu_pi1[dt]=" + "0"+ str(y_n.day)+ ".0" + str(y_n.month) + "." + str(y_n.year) + "&tx_ms1cdu_pi1[format]=csv"
    elif y_n.day<10:
        url = "http://so-ups.ru/index.php?id=oes_east_gen_consump_hour&tx_ms1cdu_pi1[dt]=" + "0"+ str(y_n.day)+ "." + str(y_n.month) + "." + str(y_n.year) + "&tx_ms1cdu_pi1[format]=csv"
    filename = str(y_n.day) + "." + str(y_n.month) + "." + str(y_n.year) + ".csv"
    r = requests.get(url, verify=False)
    with open(filename, 'wb') as f:
        f.write(r.content)
    print("Now connected to " + url + " | writing to " + filename)
    y_n=y_n+diff
