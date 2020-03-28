import _pickle as pickle
#print(my_random_forest)



# print(my_random_forest.predict([[5.84,3.0,3.75,1.1]]))

class predictIris:
	def __init__(self,sl,sw,pl,pw):
		self.sl = sl
		self.sw = sw
		self.pl = pl
		self.pw = pw
	def get_predict(self):
		my_random_forest = pickle.load(open("../../iris_rfc.pkl","rb"))
		return my_random_forest.predict([[self.sl,self.sw,self.pl,self.pw]])[0]

# ir = predictIris(5.84,3.0,3.75,1.1)
# a = ir.get_predict()
# print(a)

