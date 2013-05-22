/*
 * PosteriorHDF5Writer.cpp
 *
 *  Created on: Aug 2, 2012
 *      Author: stsiab
 */

#include <vector>
#include <cstring>

#include "types.hpp"
#include "PosteriorHDF5Writer.hpp"

namespace EpiRisk {

PosteriorHDF5Writer::PosteriorHDF5Writer(std::string filename,
		GpuLikelihood& likelihood) :
  PosteriorWriter(likelihood), isFirstWrite_(true), file_(NULL) {
	// Open HDF5 file here
	file_ = new H5::H5File(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if(!file_) {
	  stringstream msg;
	  msg << "Cannot open file '" << filename << "' in " << __PRETTY_FUNCTION__;
	  throw runtime_error(msg.str().c_str());
	}
	// Create posterior group
	H5::Group posterior(file_->createGroup("/posterior"));

	file_->flush(H5F_SCOPE_GLOBAL);

}

PosteriorHDF5Writer::~PosteriorHDF5Writer() {
	if (!isFirstWrite_) {
		delete paramTable_;
		delete infecTable_;
	}
	file_->close();
	delete file_;
}

void PosteriorHDF5Writer::write() {
	if (isFirstWrite_) {

		// Create parameter dataspace
		H5::Group posterior(file_->openGroup("/posterior"));
		hsize_t dim = paramTags_.size();
		H5::ArrayType paramRow_t(H5::PredType::NATIVE_FLOAT, 1, &dim);
		paramTable_ = new FL_PacketTable(posterior.getId(), "parameters",
				paramRow_t.getId(), PARAMCHUNK);


		// Create infection posteriors group
		H5::CompType infecTuple_t(sizeof(IPTuple_t));
		infecTuple_t.insertMember("idx", HOFFSET(IPTuple_t, idx), H5::PredType::NATIVE_INT);
		infecTuple_t.insertMember("val", HOFFSET(IPTuple_t, val), H5::PredType::NATIVE_FLOAT);
		H5::VarLenType infecRow_t(&infecTuple_t);
		infecTable_ = new FL_PacketTable(posterior.getId(), "infections",
				infecRow_t.getId(), INFECCHUNK);

		// Parameter tag attributes
		hsize_t paramAttrDims(paramTags_.size());
		H5::DataSpace* paramAttrSpace = new H5::DataSpace(1, &paramAttrDims);
		H5::StrType paramTag_t(H5::PredType::C_S1, H5T_VARIABLE);
		//paramTag_t.setSize(H5T_VARIABLE );
		H5::Attribute paramAttr(
				file_->openDataSet("/posterior/parameters").createAttribute(
						"tags", paramTag_t, *paramAttrSpace));

		char** tags = new char*[paramTags_.size()];
		for (int i = 0; i < paramTags_.size(); ++i) {
		  cout << "Tag: " << paramTags_[i].c_str() << endl;
		  tags[i] = new char[paramTags_[i].length() + 1];
		  strcpy(tags[i], paramTags_[i].c_str());
		}
		paramAttr.write(paramTag_t, tags);

		for(int i = 0; i < paramTags_.size(); ++i) delete[] tags[i];
		delete[] tags;

		// Farm IDs
		hsize_t idDim(likelihood_.GetPopulationSize());
		H5::DataSpace idSpace(1, &idDim);
		H5::DataSet idSet = posterior.createDataSet("ids", paramTag_t, idSpace);

		char** ids = new char*[likelihood_.GetPopulationSize()];
		std::vector<std::string> idVec;
		likelihood_.GetIds(idVec);
		for (int i = 0; i < likelihood_.GetPopulationSize(); ++i) {
		  ids[i] = new char[idVec[i].length() + 1];
		  strcpy(ids[i], idVec[i].c_str());
		}

		idSet.write(ids, paramTag_t);
		for(int i = 0; i < likelihood_.GetPopulationSize(); ++i) delete[] ids[i];
		delete[] ids;

		// Clean up
		delete paramAttrSpace;
		

		// Initialise parameter array
		valueBuff_.resize(paramTags_.size());

		isFirstWrite_ = false;

		file_->flush(H5F_SCOPE_GLOBAL);
	}

	// Write params
	for (size_t i = 0; i < paramVals_.size(); ++i)
		valueBuff_[i] = paramVals_[i]->GetValue();
	for (size_t i = 0; i < special_.size(); ++i)
		valueBuff_[i + paramVals_.size()] = special_[i]();
	paramTable_->AppendPacket(valueBuff_.data());

	// Write infecs
	hvl_t buff;
	std::vector<IPTuple_t> vBuff;
	likelihood_.GetInfectiousPeriods(vBuff);
	buff.len = vBuff.size();
	buff.p = vBuff.data();
	infecTable_->AppendPacket(&buff);

	// Flush
	//file_->flush(H5F_SCOPE_GLOBAL);
}

void PosteriorHDF5Writer::flush() {
	if (!isFirstWrite_)
		file_->flush(H5F_SCOPE_LOCAL);
}

} /* namespace EpiRisk */
