//
// Created by daniellopez on 6/12/18.
//

#ifndef NEURALNETS_CAPSULENETWORKDAO_H
#define NEURALNETS_CAPSULENETWORKDAO_H

#include <pqxx/pqxx>
#include <string>
#include <CapsNetConfig.h>
#include "Individual.h"

using namespace std;
using namespace pqxx;

class CapsNetDAO {
public:
    static CapsNetDAO* getInstance();
    bool isInDatabase(const Individual& bitstring);
    void getFromDatabase(Individual& individual);
    void addToDatabase(Individual& individual);

private:
    CapsNetDAO() = default;
    static CapsNetDAO* instance;
    void run_sql(const string& sql, result& output);
    void commit_sql(const string& sql, result& output);
    string host;
};


#endif //NEURALNETS_CAPSULENETWORKDAO_H
