class Student: 
	def __init__(self, major=None, classification=None, email=None, name=None, phone=None): 
		self.name = name
		self.major = major
		self.classification = classification
		self.phone=phone
		self.email = email
		self.affiliation = None
	def __str__(self): 
		return "{} \n {} \n {} \n {} \n {} \n\n".format(self.name, self.major, self.classification, self.phone, self.email)
