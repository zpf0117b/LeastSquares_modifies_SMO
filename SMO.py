# coding=utf-8
from random import randint
import numpy as np
from math import tanh, inf
from heapq import nsmallest
from time import process_time

# print a.astype(int)
# print a.A

class SVM(object):
	# data:collections of [vector(x),y]
	def __init__(self, C=100, epsilon=0.00001, optimize=True, kernel=('linear', {'sigma':None, 'offset':None})):
		self.C = C  # 惩罚因子
		self.kernel = kernel
		self.Iter = [0, 0]
		self.epsilon = epsilon
		self.eCache = None
		self.xL, self.yL, self.a, self.w, self.b = None, None, None, None, None
		self.optimize = optimize
		self.rownum = None
		self.columnnum = None

	def fit(self, train_x, train_y):
		self.xL = np.array(train_x.copy())
		self.yL = np.array(train_y.copy())
		self.rownum, self.columnnum = np.shape(self.xL)
		self.a = self.initial_a(self.optimize)
		self.w = np.zeros((self.columnnum))
		self.b = 0
		self.eCache = [[False, 0] for i in range(self.rownum)]
		self.SMO()

	def initial_a(self, optimize=True):
		alpha = np.zeros((self.rownum)).ravel()
		if self.optimize:
			bb = 0
			ww = np.zeros((self.columnnum)).ravel()
			for i in range(self.rownum):
				bb += np.linalg.norm(self.xL[i]) / self.rownum
			for i in range(self.columnnum):
				ww[i] = - 2 *bb
				for j in range(self.rownum):
					ww[i] += self.xL[j][i] / self.rownum
			distance = {}
			for i in range(self.rownum):
				distance[i] = abs(ww @ self.xL[i] + bb)/np.linalg.norm(ww)
			ns = nsmallest(5, zip(distance.values(), distance.keys()))
			Xmatrix = np.array([
			[self.xL[ns[0][1], 0],self.xL[ns[0][1],1],self.xL[ns[0][1],2],self.xL[ns[0][1],3],self.xL[ns[0][1],4]],
			[self.xL[ns[1][1], 0],self.xL[ns[1][1],1],self.xL[ns[1][1],2],self.xL[ns[1][1],3],self.xL[ns[1][1],4]],
			[self.xL[ns[2][1], 0],self.xL[ns[2][1],1],self.xL[ns[2][1],2],self.xL[ns[2][1],3],self.xL[ns[2][1],4]],
			[self.xL[ns[3][1], 0],self.xL[ns[3][1],1],self.xL[ns[3][1],2],self.xL[ns[3][1],3],self.xL[ns[3][1],4]],
			[self.xL[ns[4][1], 0], self.xL[ns[4][1], 1], self.xL[ns[4][1], 2], self.xL[ns[4][1], 3], self.xL[ns[4][1], 4]]
			])
			wmatrix = np.array([ww[0], ww[1], ww[2], ww[3], ww[4]])
			alpha_matrix = wmatrix @ np.linalg.inv(Xmatrix)
			alpha_list = alpha_matrix.tolist()
			alpha_matrix /= sum(abs(alpha_list[i]) for i in range(5))/self.C
			# print(alpha_matrix)
			for i in range(5):
				alpha[ns[i][1]] = abs(alpha_matrix[i] / self.yL[ns[i][1]])
				if alpha[ns[i][1]] > self.C:
					alpha[ns[i][1]] = self.C
				elif alpha[ns[i][1]] < 0:
					alpha[ns[i][1]] = 0
			# print(alpha)
			return alpha
		else:
			return alpha

	def __kernel(self, A, B):
		kernel_dict = {'linear': lambda x, y: x @ y.T,
			'poly': lambda x, y: (self.kernel[1]['offset'] + x @ y.T) ** self.kernel[1]['sigma'],
			'rbf': lambda x, y: np.exp(-np.sqrt(np.linalg.norm(x - y) ** 2 / (2 * self.kernel[1]['sigma'] ** 2))),
			'sigmoid': lambda x, y: tanh(self.kernel[1]['sigma'] * x @ y.T + self.kernel[1]['offset'])}
		return kernel_dict[self.kernel[0]](A, B)

	def SMO(self):
		# store err cache
		for i in range(self.rownum):
			Ei = self.__calE(self.xL[i], self.yL[i])
			self.eCache[i][1] = Ei
		# print(self.eCache, self.rownum)
		terminate = True
		while terminate:
			# print(self.a)
			self.Iter[0] += 1
			for i in range(self.rownum):
				# if self.Iter[0] > 1 and (self.a[i] <= 0 or self.a[i] >= self.C):
				# 	continue
				if self.Iter[0] < 1 or (((self.yL[i]*self.eCache[i][1] < -self.epsilon) and (self.a[i]<self.C)) or ((self.yL[i]*self.eCache[i][1] >self.epsilon) and (self.a[i]>0))):
					Ei = self.eCache[i][1]
					# select J
					# get backup j from ecache, which is built for solving initially selecting
					chooselist = []
					for p in range(self.rownum):
						if self.eCache[p][0] and p != i:
							chooselist.append(p)
					if len(chooselist) > 1:
						delta_E, maxE, j, Ej = -1, -1, -1, 0
						for k in chooselist:
							if k != i:
								Ek = self.eCache[k][1]
								delta_E = abs(Ei - Ek)
								if delta_E > maxE and Ek != Ei:
									maxE, j, Ej = delta_E, k, Ek
					elif len(chooselist) == 1:
						j = chooselist[0]
						Ej = self.eCache[j][1]
					else:
						j = i
						while j == i or self.eCache[j][1] == Ei:
							j = randint(0, self.rownum - 1)
						Ej = self.eCache[j][1]

					self.Iter[1] += 1
					# calculate L H
					L, H = self.__calLH(j, i)
					# transfer to a function with self.a[j] as the only variable, then differ
					kij = self.__kernel(self.xL[i], self.xL[i]) + self.__kernel(self.xL[j], self.xL[j]) - 2 * self.__kernel(self.xL[i], self.xL[j])
					if kij == 0 or L == H:
						# print('ignore', i, j, kij, L, H)
					# avoid devision by zero
						continue
					# print('notignore', i, j, kij, L, H)
					preai, preaj = self.a[i], self.a[j]
					self.a[j] += 1.0 * self.yL[j] * (Ei - Ej) / kij
					# The lower bound of a2, L=a2-a1, as intercept, will be assigned to 0 when L < 0
					# The upper bound, H=C-a1+a2, as maximum, will be assigned to C when H > C
					if self.a[j] > H and self.a[j] >= L:
						# print('H:', H, self.a[j])
						# print(Ei, Ej, preaj)
						self.a[j] = H
					elif self.a[j] < L:
						# print('L:', L, self.a[j])
						# print(Ei, Ej, preaj)
						self.a[j] = L
					self.a[i] += self.yL[i] * self.yL[j] * (preaj - self.a[j])

					# self.w self.b
					self.w = np.zeros((self.columnnum))
					for ii in range(self.rownum):
						self.w += self.a[ii] * self.yL[ii] * self.xL[ii]
					b1 = self.b - self.eCache[i][1] - self.yL[i] * (self.a[i] - preai) * self.__kernel(self.xL[i, :], self.xL[i, :]) - \
											self.yL[j] * (self.a[j] - preaj) * self.__kernel(self.xL[i, :], self.xL[j, :])

					b2 = self.b - self.eCache[j][1] - self.yL[i] * (self.a[i] - preai) * self.__kernel(self.xL[i, :], self.xL[j, :]) - \
											self.yL[j] * (self.a[j] - preaj) * self.__kernel(self.xL[j, :], self.xL[j, :])
					if (0 < self.a[i]) and (self.C > self.a[i]):  # 判断符合条件的b
						self.b = b1
					elif (0 < self.a[j]) and (self.C > self.a[j]):
						self.b = b2
					else:
						self.b = (b1 + b2) / 2.0

					self.eCache[j] = [True, self.__calE(self.xL[j], self.yL[j])]
					self.eCache[i] = [True, self.__calE(self.xL[i], self.yL[i])]

					# print(self.a[i], self.a[j])
					if self.Iter[0] > 1 and abs(preai - self.a[i]) + abs(preaj - self.a[j]) < self.epsilon:
						terminate = False
						break
			# if self.Iter % 1000 == 1 or self.Iter % 1000 == 2:
			# print(self.w, self.xL, self.rownum)
			# if (self.examine() and not (np.array(self.a) == 0).all()) or terminate:
			# 	# print('examine2')
			# 	break

	def __calLH(self,j,i):
		if(self.yL[j]!= self.yL[i]):
			return (max(0,self.a[j]-self.a[i]),min(self.C,self.C-self.a[i]+self.a[j]))
		else:
			return (max(0,-self.C+self.a[i]+self.a[j]),min(self.C,self.a[i]+self.a[j]))

	def __calE(self, x, y):
		pre_value = 0
		# 从trainData 改成 suport_Vector
		for i in range(self.rownum):
			pre_value += self.a[i] * self.yL[i] * self.__kernel(self.xL[i], x)
		pre_value += self.b
		# print pre_value,"pre_value"
		return pre_value - y

if __name__ == "__main__":

	def testing(x, y):
		test = SVM(C=1, epsilon=1e-5, optimize=True, kernel=('linear', {'sigma': None, 'offset': None}))
		y1 = y.copy().tolist()
		y2 = [-1 if y1[i] == 0 else 1 for i in range(len(y1))]
		test.fit(x, np.array(y2))
		print('iter',test.Iter)
		# print('w,b',test.w, test.b)
		# res=0
		# for i in range(test.rownum):
		# 	for j in range(test.rownum):
		# 		res += 0.5 * test.a[i] * test.a[j] * test.yL[i] * test.yL[j] * (test.xL[i] @ test.xL[j].T)
		# 	res -= test.a[i]
			# print('expect',test.a[i] * (test.yL[i] * (test.w @ test.xL[i].T + test.b - 1)))
		# print('res',res)

	x = np.random.random((5000, 20))
	# x = np.array([[-2.3,1.5],[3.4, -5.8],[-4, -4],[2, 2],[-0.8, 1.6]])
	y = np.random.randint(low=0, high=2, size=(1, 5000)).ravel()
	# print(x, y)
	# y = np.array([0,0,0,1,1])
	start = process_time()
	for i in range(1):
		testing(x, y)
	print('running time: %f s'% (process_time()-start))


