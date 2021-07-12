import math, random
import numpy as np
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation

class Sector:
    def __init__(self, division_size = 8):
        self.division_size = division_size
        self.sectors= [[] for _ in range(self.division_size * self.division_size)]

    def getBoundary(self, x, y, width=100):
        boundary = [0]
        for i in range(self.division_size) :
            boundary.append(boundary[-1] + width / self.division_size)
        xboundary = 0
        yboundary = 0
        for i in range(1, self.division_size + 1):
            if boundary[i - 1] <= x < boundary[i]:
                xboundary = i - 1
            if boundary[i - 1] <= y < boundary[i]:
                yboundary = i - 1
        return xboundary, yboundary

    def divide(self, coursemanager):
        for city in coursemanager.destinationCities:
            x = city.x
            y = city.y
            index = city.index
            xidx, yidx = self.getBoundary(x, y)
            self.sectors[yidx * self.division_size + xidx].append(index)

    def getSectors(self):
        return self.sectors

class City:
    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index
   
    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distanceTo(self, city):
        xDistance = abs(self.getX() - city.getX())
        yDistance = abs(self.getY() - city.getY())
        distance = math.sqrt( (xDistance*xDistance) + (yDistance*yDistance) )
        return distance

    def __repr__(self):
        return str(self.index)


class Manager:
    destinationCities = []

    def addCity(self, city):
        self.destinationCities.append(city)

    def getCity(self, index):
        return self.destinationCities[index]

    def numberOfCities(self):
        return len(self.destinationCities)
        
    def loadData(self, file_path, delimiter):
        i=0
        data = np.loadtxt(file_path, delimiter=delimiter)
        for line in data:
            self.addCity(City(x = line[0], y = line[1], index = i))
            i += 1
    

class Course:
    def __init__(self, course=None):
        self.fitness = 0.0
        self.distance = 0
        if course is None:
            self.course = []
        else:
            self.course = course

    def __len__(self):
        return len(self.course)

    def __getitem__(self, index):
        return self.course[index]

    def __setitem__(self, key, value):
        self.course[key] = value

    def __repr__(self):
        geneString = 'Start -> '
        for i in range(0, self.courseSize()):
            geneString += str(self.getCity(self.course[i])) + ' -> '
        geneString += str(self.getCity(self.course[0])) + ' -> '
        geneString += 'End'
        return geneString

    def getCity(self, index):
        return coursemanager.destinationCities[index]

    def setCity(self, index, cityIndex):
        self.course[index] = cityIndex
        self.fitness = 0.0
        self.distance = 0

    def extend(self, course):
        self.course.extend(course)

    def getFitness(self):
        if self.fitness == 0:
            self.fitness = 1/float(self.getDistance())
        return self.fitness

    def getDistance(self):
        if self.distance == 0:
            courseDistance = 0
            lastCity = self.getCity(0)
            for index in self.course:
                courseDistance += self.getCity(index).distanceTo(lastCity)
                lastCity = self.getCity(index)
            courseDistance += lastCity.distanceTo(self.getCity(0))
            self.distance = courseDistance
        return self.distance

    def courseExchange(self, index1, index2):
        temp = self.course[index1]
        self.course[index1] = self.course[index2]
        self.course[index2] = temp

    def courseSize(self):
        return len(self.course)

    def containsCity(self, city):
        return city in self.course

    def getIndex(self, index):
        return self.course[index]
    
    def shuffle(self):
        if self.courseSize() == 0:
            idx = random.sample(list(range(coursemanager.numberOfCities())), coursemanager.numberOfCities()) 
        else : 
            idx = random.sample(self.course, self.courseSize())
        self.course = idx

class Population:
    def __init__(self, populationSize, initialise, course=None):
        self.courses = []
        if initialise:
            newcourse = Course(course)
            newcourse.shuffle()
            self.courses.append(newcourse)
            for idx in range(populationSize-1):
                newcourse = Course(course)
                newcourse.shuffle()
                self.courses.append(newcourse)
    def __setitem__(self, key, value):
        self.courses[key] = value

    def __getitem__(self, index):
        return self.courses[index]

    def savecourse(self, index, course):
        self.courses[index] = course

    def extentioncourse(self, pop):
        self.courses.extend(pop.courses)

    def pushCourse(self, course):
        self.courses.append(course)

    def getCourse(self, index):
        return self.courses[index]

    def getFittest(self):
        self.sortByFit()
        return self.courses[0]

    def populationSize(self):
        return len(self.courses)

    def sortByFit(self):
        self.courses = sorted(self.courses, key = lambda x : (x.getFitness()), reverse=True)

class GA:
    def __init__(self, mutationRate=0.1):
        self.mutationRate = mutationRate

    def evolvePopulation(self, pop):
        newPopulation = Population(populationSize = pop.populationSize(), initialise=False)

        for i in range(pop.populationSize()-1):
            parent1 = pop.courses[i]
            parent2 = pop.courses[i+1]
            child = self.crossover(parent1, parent2)
            newPopulation.pushCourse(child)
        
        for i in range(int(pop.populationSize()/2)):
            parent1 = pop.courses[i]
            parent2 = pop.courses[-i]
            child = self.crossover(parent1, parent2)
            newPopulation.pushCourse(child)

        for i in range(int(pop.populationSize()/10)):
            newPopulation.pushCourse(pop[i])

        newPopulation.sortByFit()
        for i in range(1, newPopulation.populationSize()):
            self.mutate(newPopulation[i])

        newPopulation = self.selection(newPopulation)
        return newPopulation

    def makeTable(self, parent1, parent2):
        edge_table = {}
        for idx in range(parent1.courseSize()):
            neighbor = []
            key = parent1[idx]
            if idx == parent1.courseSize()-1:
                neighbor.append(parent1[0])
            else:
                neighbor.append(parent1[idx+1])

            if idx == 0:
                neighbor.append(parent1[-1])
            else:
                neighbor.append(parent1[idx-1])
            if key in edge_table:
                edge_table[key] += neighbor
            else :
                edge_table[key] = neighbor

            neighbor = []
            key = parent2[idx]

            if idx == parent1.courseSize()-1:
                neighbor.append(parent2[0])
            else:
                neighbor.append(parent2[idx+1])
                
            if idx == 0:
                neighbor.append(parent2[-1])
            else:
                neighbor.append(parent2[idx-1])
            if key in edge_table:
                edge_table[key] += neighbor
            else :
                edge_table[key] = neighbor
                
        return edge_table

    def crossover(self, parent1, parent2):
        child = []
        possible_set = set(parent1.course)
        neighbor_table = self.makeTable(parent1, parent2)
        node = parent1[0]
        possible_set.remove(node)
        child.append(node)
        for idx in range(parent1.courseSize()-1):
            last = child[-1]
            least = 5
            for item in neighbor_table[last]:
                cnt = 0
                if item in child:
                    continue
                for i in neighbor_table[item]:
                    if i in child:
                        continue
                    cnt+=1
                if cnt < least:
                    node = item
                    least = cnt
            if node in child:
                city = coursemanager.getCity(node)
                least_d = 10000.0
                for idx in list(possible_set):
                    if idx == node:
                        continue    
                    dist = city.distanceTo(coursemanager.getCity(idx))
                    if dist < least_d:
                        node = idx
            child.append(node)
            possible_set.remove(node)

        return Course(child)
   
    def mutate(self, course):
        if random.random() < self.mutationRate:
            course
            numberChange = int(course.courseSize()*self.mutationRate)
            if numberChange > course.courseSize(): numberChange = course.courseSize()
            last = 0
            least = course.getDistance()
            edgeList = list()
            for now in range(1, course.courseSize()):
                last_city = course.getCity(now-1)
                now_city = course.getCity(now)
                dist = last_city.distanceTo(now_city)
                edgeList.append([last_city.index, now_city.index, dist])
            edgeList = sorted(edgeList, key = lambda x : x[2], reverse=True)
            for edge in edgeList[numberChange:]:
                course.courseExchange(edge[0], edge[1])

    def selection(self, pop):
        pop.sortByFit()
        del pop.courses[population_size:]
        return pop

    def setRate(self, rate):
        self.mutationRate = rate
    
    def upRate(self, rate = 0.002):
        self.mutationRate += rate
    
    def downRate(self, rate = 0.002):
        self.mutationRate -= rate

class Graph:
    def __init__(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(list(range(coursemanager.numberOfCities())))
        self.pos = {}
        for city in coursemanager.destinationCities:
            self.pos[city.index] = [city.x, city.y]

    def draw(self, course, gene, dist):
        ax.clear()
        edges = list(self.G.edges)
        self.G.remove_edges_from(edges)
        for i in range(course.courseSize()-1):
            fr = course.getIndex(i)
            to = course.getIndex(i+1)
            self.G.add_edge(fr, to)
        self.G.add_edge(course.getIndex(course.courseSize()-1), course.getIndex(0))
        nx.draw_networkx_nodes(self.G, pos=self.pos, node_size=10)
        nx.draw_networkx_edges(self.G, pos=self.pos, ax=ax)

        ax.set_title(str(gene)+"'s G, distance: %f" %dist, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.show()
        plt.pause(0.01)
    

def solve(division_size = 8):
    sector_size = division_size*division_size
    sector = Sector(division_size=division_size)
    sector.divide(coursemanager)
    fittests = [None]*sector_size
    mergedList = []
    ga = GA()
    courses = sector.getSectors()
    for idx in range(len(courses)) :
        print('start %d sector'%idx)
        pop = Population(populationSize=population_size, initialise=True, course = courses[idx])
        last=pop.getFittest()
        for i in range(int(n_generations*(len(courses[idx])/int(coursemanager.numberOfCities()/sector_size)))):
            pop = ga.evolvePopulation(pop)
            fittests[idx] = pop.getFittest()
            if last.getFitness() == fittests[idx].getFitness(): 
                ga.upRate()
            else : 
                if ga.mutationRate > 0.3:
                    ga.setRate(0.3)
                elif ga.mutationRate > 0.1:
                    ga.downRate()
            last = fittests[idx]
            if ga.mutationRate > 0.5 :
                break
        mergedList.append(fittests[idx].course)
        ga.setRate(0.1)
        del pop

    df = pd.DataFrame(mergedList)
    df.to_csv('TSP_sol.csv', index=False, header=None)
    return mergedList

#test code
def main(division_size = 8):
    plt.ion()
    grp = Graph()
    sector_size = division_size*division_size
    sector = Sector(division_size=division_size)
    sector.divide(coursemanager)
    fittests = [None]*sector_size
    merged = Course()
    mergedList = []
    dist = []
    ga = GA()
    courses = sector.getSectors()
    for idx in range(len(courses)) :
        pop = Population(populationSize=population_size, initialise=True, course = courses[idx])
        last=pop.getFittest()
        print('start %d sector'%idx)
        for i in range(int(n_generations*(len(courses[idx])/int(coursemanager.numberOfCities()/sector_size)))):
            pop = ga.evolvePopulation(pop)
            fittests[idx] = pop.getFittest()
            print('%d세대, 거리: %s'%(i,fittests[idx].getDistance()))
            print('변이 확률: %f'%ga.mutationRate)
            if last.getFitness() == fittests[idx].getFitness(): 
                ga.upRate()
            else : 
                if ga.mutationRate > 0.3:
                    ga.setRate(0.3)
                elif ga.mutationRate > 0.1:
                    ga.downRate()
            last = fittests[idx]
            if ga.mutationRate > 0.5 :
                print('%dG' %i)
                break
        grp.draw(fittests[idx], idx, fittests[idx].getDistance())
        dist.append(fittests[idx].getDistance())
        merged.extend(fittests[idx])
        mergedList.append(fittests[idx].course)
        ga.setRate(0.1)
        del pop

    grp.draw(merged, 0, sum(dist))
    plt.savefig('save_TSP.png', dpi=400)
    print(mergedList)
    df = pd.DataFrame(mergedList)
    df.to_csv('TSP_sol.csv', index=False, header=None)
    print("Finished")
   
global coursemanager
coursemanager = Manager()
coursemanager.loadData(file_path='./TSP.csv', delimiter=',')
global ax, fig
fig, ax = plt.subplots(figsize=(8, 8))

population_size = 500
n_generations = 100

#test code
if __name__ == '__main__':
    main(8)