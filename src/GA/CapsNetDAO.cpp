//
// Created by daniellopez on 6/12/18.
//

#include <iostream>
#include <thread>
#include "GA/CapsNetDAO.h"

CapsNetDAO* CapsNetDAO::instance = nullptr;

CapsNetDAO* CapsNetDAO::getInstance() {
    if (instance == nullptr) {
        instance = new CapsNetDAO;
    }
    return instance;
}

bool CapsNetDAO::isInDatabase(const Individual &bitstring) {
    string query = "select count(bitstring) from config where bitstring = '" + bitstring.to_string() + "'";
    result row;

    run_sql(query, row);

    return row[0][0].as<int>() > 0;
}

void CapsNetDAO::getFromDatabase(Individual &individual) {
    string query = "select * from config where bitstring = '" + individual.to_string() + "';";
    result row;
    run_sql(query, row);

    individual.accuracy_100 = row[0]["accuracy_100"].as<double>();
    individual.accuracy_300 = row[0]["accuracy_300"].as<double>();
    individual.loss_100 = row[0]["loss_100"].as<double>();
    individual.loss_300 = row[0]["loss_300"].as<double>();
}

void CapsNetDAO::addToDatabase(Individual &individual) {
    string query = "insert into public.config ("
            "bitstring, m_plus, m_minus, lambda, num_tensor_channels, "
            "batch_size, innerdim, outerdim, accuracy_100, accuracy_300, loss_100, loss_300) ";
    stringstream values;
    values << "VALUES (" << "'" << individual.to_string() << "'" << ", ";
    values << individual.capsNetConfig.m_plus << ", ";
    values << individual.capsNetConfig.m_minus << ", ";
    values << individual.capsNetConfig.lambda << ", ";
    values << individual.capsNetConfig.cnNumTensorChannels << ", ";
    values << individual.capsNetConfig.batchSize << ", ";
    values << individual.capsNetConfig.cnInnerDim << ", ";
    values << individual.capsNetConfig.cnOuterDim << ", ";
    values << individual.accuracy_100 << ", ";
    values << individual.accuracy_300 << ", ";
    values << individual.loss_100 << ", ";
    values << individual.loss_300;
    values << ");";

    query += values.str();

    result row;
    commit_sql(query, row);

}

void CapsNetDAO::run_sql(const string &sql, result &output) {
    connection c("dbname=cs_776 user=system password=SYSTEM host=hpcvis3.cse.unr.edu");
    work txn(c);
    output = txn.exec(sql);
}

void CapsNetDAO::commit_sql(const string &sql, result &output) {
    connection c("dbname=cs_776 user=system password=SYSTEM host=hpcvis3.cse.unr.edu");
    work txn(c);

    output = txn.exec(sql);
    txn.commit();
}