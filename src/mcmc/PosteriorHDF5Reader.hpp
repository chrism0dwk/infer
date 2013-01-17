/*
 * PosteriorHDF5Reader.hpp
 *
 *  Created on: Dec 17, 2012
 *      Author: cpjewell
 */

#ifndef POSTERIORHDF5READER_HPP_
#define POSTERIORHDF5READER_HPP_

#include <string>
#include <vector>

#include <H5Cpp.h>
#include <H5PacketTable.h>

#include "types.hpp"

namespace EpiRisk {

#define PARAMPATH "posterior/parameters"
#define INFECPATH "posterior/infections"
#define IDPATH "posterior/ids"

	class PosteriorHDF5Reader {
	public:
		PosteriorHDF5Reader(const std::string filename);
		PosteriorHDF5Reader(const PosteriorHDF5Reader& other);
		virtual ~PosteriorHDF5Reader();

		void
		GetParameterTags(std::vector<std::string>& tags);
		void
		GetPopulationIds(std::vector<std::string>& ids);

		void
		GetParameters(const size_t idx, std::vector<fp_t>& parameters);
		void
		GetInfections(const size_t idx, std::vector<IPTuple_t>& ips);

		size_t
		GetSize() const;

	private:

		struct IPTupleHack_t
		{
		  int idx;
		  float val;
		};

		std::string filename_;
		H5::H5File* theFile_;
		H5::DataSet parameters_;
		H5::DataSet infections_;
		FL_PacketTable* parametersPt_;
		FL_PacketTable* infectionsPt_;
		size_t size_;
		size_t numParams_;

	};

} /* namespace EpiRisk */
#endif /* POSTERIORHDF5READER_HPP_ */
