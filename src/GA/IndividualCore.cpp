// Host-testable genetic-algorithm individual semantics.

#include "GA/Individual.h"

#include <Utils.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

constexpr std::size_t kGeneWidth = 5;
constexpr std::size_t kGeneCount = 7;
constexpr std::size_t kChromosomeSize = kGeneWidth * kGeneCount;

int decodeGene(const Individual& individual, std::size_t geneIndex) {
    const auto first = individual.begin() + static_cast<std::ptrdiff_t>(geneIndex * kGeneWidth);
    return Utils::getBinaryAsInt(std::vector<bool>(first, first + static_cast<std::ptrdiff_t>(kGeneWidth)));
}

}  // namespace

Individual::Individual(int bitstringSize, const string& chromosome)
        : loss_100(0.0),
          loss_300(0.0),
          accuracy_100(0.0),
          accuracy_300(0.0),
          crowdingDistance(0.0),
          rank(0) {
    if (bitstringSize <= 0) {
        throw std::invalid_argument("bitstringSize must be positive");
    }

    resize(static_cast<std::size_t>(bitstringSize));
    if (chromosome.empty()) {
        generateRandom();
    } else {
        if (chromosome.size() != static_cast<std::size_t>(bitstringSize)) {
            throw std::invalid_argument("chromosome size does not match bitstringSize");
        }
        for (std::size_t i = 0; i < chromosome.size(); ++i) {
            if (chromosome[i] != '0' && chromosome[i] != '1') {
                throw std::invalid_argument("chromosome may contain only '0' and '1'");
            }
            (*this)[i] = chromosome[i] == '1';
        }
    }
    decodeChromosome();
}

Individual::Individual(const Individual& src)
        : vector<bool>(src),
          capsNetConfig(src.capsNetConfig),
          loss_100(src.loss_100),
          loss_300(src.loss_300),
          accuracy_100(src.accuracy_100),
          accuracy_300(src.accuracy_300),
          crowdingDistance(src.crowdingDistance),
          rank(src.rank),
          numDominateMe(0) {
    // Dominance pointers are transient graph state. Copying them would leave a
    // new Individual pointing into the source population.
    individualsIDominate.clear();
}

void Individual::print() const {
    cout << to_string() << endl;
}

void Individual::fullPrint() const {
    string chromosome = to_string();
    if (chromosome.size() == kChromosomeSize) {
        for (std::size_t offset = kGeneWidth; offset < chromosome.size(); offset += kGeneWidth + 1) {
            chromosome.insert(offset, 1, ' ');
        }
    }

    cout << chromosome << ": " << endl;
    cout << "         cnInnerDim: " << capsNetConfig.cnInnerDim << endl;
    cout << "         cnOuterDim: " << capsNetConfig.cnOuterDim << endl;
    cout << "cnNumTensorChannels: " << capsNetConfig.cnNumTensorChannels << endl;
    cout << "          batchSize: " << capsNetConfig.batchSize << endl;
    cout << "             m_plus: " << capsNetConfig.m_plus << endl;
    cout << "            m_minus: " << capsNetConfig.m_minus << endl;
    cout << "             lambda: " << capsNetConfig.lambda << endl;
    cout << endl;
    cout << "     Accuracy (100): " << accuracy_100 << endl;
    cout << "         Loss (100): " << loss_100 << endl;
    cout << "     Accuracy (300): " << accuracy_300 << endl;
    cout << "          Loss(300): " << loss_300 << endl;
}

string Individual::to_string() const {
    string result;
    result.reserve(size());
    for (bool bit : *this) {
        result += bit ? '1' : '0';
    }
    return result;
}

void Individual::generateRandom() {
    for (std::size_t i = 0; i < size(); ++i) {
        (*this)[i] = Utils::randomWithProbability(0.5);
    }
}

bool Individual::operator==(const Individual &other) const {
    if (size() != other.size()) {
        return false;
    }
    return std::equal(begin(), end(), other.begin());
}

Individual& Individual::operator=(const Individual &other) {
    if (&other == this) {
        return *this;
    }

    vector<bool>::operator=(other);
    capsNetConfig = other.capsNetConfig;
    accuracy_100 = other.accuracy_100;
    accuracy_300 = other.accuracy_300;
    loss_100 = other.loss_100;
    loss_300 = other.loss_300;
    crowdingDistance = other.crowdingDistance;
    rank = other.rank;

    individualsIDominate.clear();
    numDominateMe = 0;
    return *this;
}

void Individual::decodeChromosome() {
    if (size() != kChromosomeSize) {
        throw std::invalid_argument("CapsNet chromosome must contain exactly 35 bits");
    }

    capsNetConfig.cnInnerDim = decodeGene(*this, 0) + 2;
    capsNetConfig.cnOuterDim = decodeGene(*this, 1) + 2;
    capsNetConfig.cnNumTensorChannels = decodeGene(*this, 2) + 1;
    capsNetConfig.batchSize = (decodeGene(*this, 3) + 1) * 20;
    capsNetConfig.m_plus = static_cast<double>(decodeGene(*this, 4)) / 160.0 + 0.8;
    capsNetConfig.m_minus = static_cast<double>(decodeGene(*this, 5)) / 160.0 + 0.00625;
    capsNetConfig.lambda = static_cast<double>(decodeGene(*this, 6)) / 160.0 + 0.4;
}

bool Individual::paredoDominates(const Individual &opponent) const {
    const bool noWorse =
        accuracy_100 >= opponent.accuracy_100 &&
        accuracy_300 >= opponent.accuracy_300 &&
        loss_100 <= opponent.loss_100 &&
        loss_300 <= opponent.loss_300;

    const bool strictlyBetter =
        accuracy_100 > opponent.accuracy_100 ||
        accuracy_300 > opponent.accuracy_300 ||
        loss_100 < opponent.loss_100 ||
        loss_300 < opponent.loss_300;

    return noWorse && strictlyBetter;
}

bool Individual::crowdingOperator(const Individual& opponent) const {
    if (this == &opponent) {
        return false;
    }
    if (rank != opponent.rank) {
        return rank < opponent.rank;
    }
    return crowdingDistance > opponent.crowdingDistance;
}

void Individual::crossoverWith(Individual &other) {
    if (empty() || size() != other.size()) {
        throw std::invalid_argument("crossover requires non-empty chromosomes of equal size");
    }

    const int crossoverPoint = Utils::getRandBetween(0, static_cast<int>(size()));
    for (std::size_t i = static_cast<std::size_t>(crossoverPoint); i < size(); ++i) {
        const bool temp = (*this)[i];
        (*this)[i] = other[i];
        other[i] = temp;
    }
    decodeChromosome();
    other.decodeChromosome();
}

void Individual::mutate() {
    if (empty()) {
        throw std::logic_error("cannot mutate an empty chromosome");
    }

    const int mutationPoint = Utils::getRandBetween(0, static_cast<int>(size()));
    (*this)[static_cast<std::size_t>(mutationPoint)] = !at(static_cast<std::size_t>(mutationPoint));
    decodeChromosome();
}
