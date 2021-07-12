import math, random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import pandas as pd
import random
import time

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

    def divide(self, tourmanager):
        for city in tourmanager.destinationCities:
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


class TourManager:
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
    

class Tour:
    def __init__(self, tour=None):
        self.fitness = 0.0
        self.distance = 0
        if tour is None:
            self.tour = []
        else:
            self.tour = tour

    def __len__(self):
        return len(self.tour)

    def __getitem__(self, index):
        return self.tour[index]

    def __setitem__(self, key, value):
        self.tour[key] = value

    def __repr__(self):
        geneString = 'Start -> '
        for i in range(0, self.tourSize()):
            geneString += str(self.getCity(self.tour[i])) + ' -> '
        geneString += str(self.getCity(self.tour[0])) + ' -> '
        geneString += 'End'
        return geneString

    def getCity(self, index):
        return tourmanager.destinationCities[index]

    def setCity(self, index, cityIndex):
        self.tour[index] = cityIndex
        self.fitness = 0.0
        self.distance = 0

    def extend(self, tour):
        self.tour.extend(tour)

    def getFitness(self):
        if self.fitness == 0:
            self.fitness = 1/float(self.getDistance())
        return self.fitness

    def getDistance(self):
        if self.distance == 0:
            tourDistance = 0
            lastCity = self.getCity(0)
            for index in self.tour:
                tourDistance += self.getCity(index).distanceTo(lastCity)
                lastCity = self.getCity(index)
            tourDistance += lastCity.distanceTo(self.getCity(0))
            self.distance = tourDistance
        return self.distance

    def tourExchange(self, index1, index2):
        temp = self.tour[index1]
        self.tour[index1] = self.tour[index2]
        self.tour[index2] = temp

    def tourSize(self):
        return len(self.tour)

    def containsCity(self, city):
        return city in self.tour

    def getIndex(self, index):
        return self.tour[index]
    
    def shuffle(self):
        if self.tourSize() == 0:
            idx = random.sample(list(range(tourmanager.numberOfCities())), tourmanager.numberOfCities()) 
        else : 
            idx = random.sample(self.tour, self.tourSize())
        self.tour = idx

class Population:
    def __init__(self, populationSize, initialise, tour=None):
        self.tours = []
        if initialise:
            newtour = Tour(tour)
            newtour.shuffle()
            self.tours.append(newtour)
            for idx in range(populationSize-1):
                newtour = Tour(tour)
                newtour.shuffle()
                self.tours.append(newtour)
    def __setitem__(self, key, value):
        self.tours[key] = value

    def __getitem__(self, index):
        return self.tours[index]

    def saveTour(self, index, tour):
        self.tours[index] = tour

    def extentionTour(self, pop):
        self.tours.extend(pop.tours)

    def pushTour(self, tour):
        self.tours.append(tour)

    def getTour(self, index):
        return self.tours[index]

    def getFittest(self):
        self.sortByFit()
        return self.tours[0]

    def populationSize(self):
        return len(self.tours)

    def sortByFit(self):
        self.tours = sorted(self.tours, key = lambda x : (x.getFitness()), reverse=True)

class GA:
    def __init__(self, mutationRate=0.1):
        self.mutationRate = mutationRate

    def evolvePopulation(self, pop):
        newPopulation = Population(populationSize = pop.populationSize(), initialise=False)

        for i in range(pop.populationSize()-1):
            parent1 = pop.tours[i]
            parent2 = pop.tours[i+1]
            child = self.crossover(parent1, parent2)
            newPopulation.pushTour(child)
        
        for i in range(int(pop.populationSize()/2)):
            parent1 = pop.tours[i]
            parent2 = pop.tours[-i]
            child = self.crossover(parent1, parent2)
            newPopulation.pushTour(child)

        newPopulation.saveTour(-1, pop[0])
        newPopulation.sortByFit()
        for i in range(1, newPopulation.populationSize()):
            self.mutate(newPopulation[i])

        newPopulation = self.selection(newPopulation)
        return newPopulation

    def makeTable(self, parent1, parent2):
        edge_table = {}
        for idx in range(parent1.tourSize()):
            neighbor = []
            key = parent1[idx]
            if idx == parent1.tourSize()-1:
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

            if idx == parent1.tourSize()-1:
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
        possible_set = set(parent1.tour)
        neighbor_table = self.makeTable(parent1, parent2)
        node = parent1[0]
        possible_set.remove(node)
        child.append(node)
        for idx in range(parent1.tourSize()-1):
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
                city = tourmanager.getCity(node)
                least_d = 10000.0
                for idx in list(possible_set):
                    if idx == node:
                        continue    
                    dist = city.distanceTo(tourmanager.getCity(idx))
                    if dist < least_d:
                        node = idx
            child.append(node)
            possible_set.remove(node)

        return Tour(child)
   
    def mutate(self, tour):
        if random.random() < self.mutationRate:
            numberChange = int(tour.tourSize()*self.mutationRate)
            if numberChange > tour.tourSize(): numberChange = tour.tourSize()
            last = 0
            least = tour.getDistance()
            edgeList = list()
            for now in range(1, tour.tourSize()):
                last_city = tour.getCity(now-1)
                now_city = tour.getCity(now)
                dist = last_city.distanceTo(now_city)
                edgeList.append([last_city.index, now_city.index, dist])
            edgeList = sorted(edgeList, key = lambda x : x[2], reverse=True)
            for edge in edgeList[numberChange:]:
                tour.tourExchange(edge[0], edge[1])

    def selection(self, pop):
        pop.sortByFit()
        del pop.tours[population_size:]
        return pop

    def setRate(self, rate = 0.1):
        self.mutationRate = rate
    
    def upRate(self, rate = 0.002):
        self.mutationRate += rate
    
    def downRate(self, rate = 0.002):
        self.mutationRate -= rate

class Graph:
    def __init__(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(list(range(tourmanager.numberOfCities())))
        self.pos = {}
        for city in tourmanager.destinationCities:
            self.pos[city.index] = [city.x, city.y]

    def draw(self, tour, gene, dist):
        ax.clear()
        edges = list(self.G.edges)
        self.G.remove_edges_from(edges)
        for i in range(tour.tourSize()-1):
            fr = tour.getIndex(i)
            to = tour.getIndex(i+1)
            self.G.add_edge(fr, to)
        self.G.add_edge(tour.getIndex(tour.tourSize()-1), tour.getIndex(0))
        nx.draw_networkx_nodes(self.G, pos=self.pos, node_size=10)
        nx.draw_networkx_edges(self.G, pos=self.pos, ax=ax)

        ax.set_title(str(gene)+"'s G, distance: %f" %dist, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.show()
        plt.pause(0.01)
    

def main():
    plt.ion()
    grp = Graph()
    division_size = 1
    sector_size = division_size*division_size
    sector = Sector(division_size=division_size)
    sector.divide(tourmanager)
    fittests = [None]*sector_size
    merged = Tour()
    mergedList = []
    distances = [None]*sector_size
    ga = GA()
    tours = sector.getSectors()
    for idx in range(len(tours)) :
        pop = Population(populationSize=population_size, initialise=True, tour = tours[idx])
        last=pop.getFittest()
        print('start %d sector'%idx)
        for i in range(int(n_generations*(len(tours[idx])/int(tourmanager.numberOfCities()/sector_size)))):
            pop = ga.evolvePopulation(pop)
            fittests[idx] = pop.getFittest()
            print('%d세대, 거리: %s'%(i,fittests[idx].getDistance()))
        grp.draw(fittests[idx], idx, fittests[idx].getDistance())
        merged.extend(fittests[idx])
        mergedList.append(fittests[idx].tour)
        ga.setRate(0.1)
        del pop

    grp.draw(merged, 0, merged.getDistance())
    plt.savefig('save_TSP.png', dpi=400)
    print(mergedList)
    df = pd.DataFrame(mergedList)
    df.to_csv('TSP_sol.csv', index=True)
    print("Finished")
   
global tourmanager
tourmanager = TourManager()
tourmanager.loadData(file_path='./TSP.csv', delimiter=',')
global ax, fig
fig, ax = plt.subplots(figsize=(8, 8))

population_size = 500
n_generations = 1000
if __name__ == '__main__':
    main()