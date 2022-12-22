############################
##############################
#Temel String İşlemleri
############################
##############################
###########################
#Oluşturma Biçimlendirme
###########################


isimler = ['ali', 'veli', 'ayse']

for i in isimler :
    print('_', i[0:], sep="")

##############################

for i in isimler :
    print("İsim:", i, sep="")

##############################

print(*enumerate(isimler))

##############################

for i in enumerate("isimler"):
    print(i)

##############################

for i in enumerate(isimler,4):
    print(i)

##############################
###########################
#Dizi İçi Tip Sorgulamaları
###########################
"mvk".isalpha()
###########################
"mvk30".isalpha()
###########################
"123".isnumeric()
###########################
"123".isdigit()
###########################
"mvk30".isalnum()

###########################
# Elemanlarına ve Eleman İndexlerine Erişmek
###########################
isim = 'mustafavahitkeskin'
isim[0:2]
isim.index('n')
isim.index("a")
###########################
# Başlangıç ve Bitiş Karakterlerini Sorgulamak
###########################
isim.startswith('a')
isim.startswith('M')

isim.endswith('n')
isim.count("a")

sorted('defter')
print(*sorted('defter'), sep="")

###########################
# Karakterleri Bölmek
###########################
isim = "Mustafa Vahit Keskin"
isim.split()
isim = "Mustafa_Vahit_Keskin"
isim.split("_")
isim.upper()
isim.lower()
isim.isupper()
isim.islower()

###########################
# Capitalize & Title
###########################
isim = "mustafa vahit keskin"
isim.capitalize()
isim.title()
isim = "mustafa VAHİT keskin"
isim.swapcase()

###########################
# istenmeyen karakterleri kırpmak
###########################
isim = " hello "
isim.strip()
isim= "*hello*"
isim.strip("*")
isim.lstrip("*")
isim.rstrip("*")

###########################
# Join
###########################
isim = "mustafa vahit keskin"
ayrık = isim.split()
joiner = " "
joiner.join(ayrık)
###########################
# Eleman Değiştirme
###########################
isim.replace("v", "c")

ifade = " Bu ifade içerisinde bağzı Türkçe karakterler vardır"
düzletilecek_harfler = "çÇğĞıİöÖŞşüÜ"
düzeltilmiş_harfler = "cCgGiIoOSsuU"
alfabe_düzeltme = str.maketrans(düzletilecek_harfler,düzeltilmiş_harfler)
ifade.translate(alfabe_düzeltme)

###########################
# Contains
###########################
isimler = ["ayse", "Ayşe", "ALİ", "ali"]
import pandas as pd
v= pd.Series(isimler)
v.str.contains("al")
v.str.contains("al").sum()
v.str.contains("[aA][lL]")