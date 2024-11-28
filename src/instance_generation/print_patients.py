patients = [
	["{0:4, 1:5, 2:3}",				"12","DayHour(day=0)",		"DayHour(day=6)",		"False","{}","[]"],
	["{1:4, 2:1}",					"12","DayHour(day=2)",		"DayHour(day=10)",	"False","{}","[]"],
	["{0:3, 2:3}",					"12","DayHour(day=4)",		"DayHour(day=10)",	"False","{}","[]"],
	["{1:6, 2:1, 1:1}",				"15","DayHour(day=5)",		"DayHour(day=9)",		"False","{}","[]"],
	["{0:4, 1:5, 2:3}",				"15","DayHour(day=6)",		"DayHour(day=13)",	"False","{}","[]"],
	["{1:4, 2:1}",					"15","DayHour(day=8)",		"DayHour(day=14)",	"False","{}","[]"],
	["{0:3, 2:3}",					"12","DayHour(day=10)",		"DayHour(day=14)",	"False","{}","[]"],
	["{0:4, 1:5, 2:3}",				"12","DayHour(day=13)",		"DayHour(day=19)",	"False","{}","[]"],
	["{1:4, 2:1}",					"12","DayHour(day=16)",		"DayHour(day=20)",	"False","{}","[]"],
	["{0:3, 2:3}",					"12","DayHour(day=2)",		"DayHour(day=15)",	"False","{}","[]"],
	["{1:6, 2:1, 1:1}",				"15","DayHour(day=4)",		"DayHour(day=20)",	"False","{}","[]"],
	["{0:4, 1:5, 2:3}",				"15","DayHour(day=9)",		"DayHour(day=11)",	"False","{}","[]"],
	["{1:4, 2:1}",					"15","DayHour(day=9)",		"DayHour(day=13)",	"False","{}","[]"],
	["{0:3, 2:3}",					"12","DayHour(day=11)",		"DayHour(day=12)",	"False","{}","[]"],	
	# Already admitted leaving on days 0,1,2,3,4,5
	["{0:1, 1:1, 3:1}",				"1","DayHour(day=0)",		"DayHour(day=1)",		"True","{}","[]"],
 	["{0:1, 1:1, 3:1, 4:1, 5:1, 6:1}","2","DayHour(day=0)",		"DayHour(day=1)",		"True","{}","[]"],
  	["{0:1, 1:2, 3:1, 4:2, 5:1, 6:2}","3","DayHour(day=0)",		"DayHour(day=1)",		"True","{}","[]"],
    ["{0:1, 1:2, 3:1, 4:2, 5:1, 6:2}","3","DayHour(day=0)",		"DayHour(day=1)",		"True","{}","[]"],
    ["{0:2, 1:2, 3:1, 4:2, 5:3, 6:2}","4","DayHour(day=0)",		"DayHour(day=1)",		"True","{}","[]"],
    ["{0:3, 1:2, 3:1, 4:2, 5:3, 6:3}","6","DayHour(day=0)",		"DayHour(day=1)",		"True","{}","[]"],
]

for i, p in enumerate(patients):
    s = ";".join(p)
    print(f"{i}; patient{i};{s}")
