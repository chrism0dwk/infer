#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include "Likelihood.hpp"


namespace EpiRisk
{

  float
  GetDistElement(const CsrMatrix* d, const int row, const int col) {
    assert(row < d->n);
    assert(col < d->m);
    
    int start = d->rowPtr[row];
    int end = d->rowPtr[row+1];
    for(int j = start; j<end; ++j)
      if (d->colInd[j] == col) return d->val[j];
    return EpiRisk::POSINF;
  }


  bool
  getDistMatrixElement(const int row, const int col, const CsrMatrix* csrMatrix,
      float* val)
  {
    int* cols = csrMatrix->colInd + csrMatrix->rowPtr[row];
    float* vals = csrMatrix->val + csrMatrix->rowPtr[row];
    int rowlen = csrMatrix->rowPtr[row + 1] - csrMatrix->rowPtr[row];

    for (int ptr = 0; ptr < rowlen; ++ptr)
      {
        if (cols[ptr] == col)
          {
            *val = vals[ptr];
            return true;
          }
      }
    return false;
  }


  float
  timeinseconds(const timeval a, const timeval b)
  {
    timeval result;
    timersub(&b, &a, &result);
    return result.tv_sec + result.tv_usec / 1000000.0;
  }



  Likelihood::Likelihood(PopDataImporter& population, EpiDataImporter& epidemic,
			 const size_t nSpecies, const fp_t obsTime,
			 const bool occultsOnlyDC) :
    popSize_(0),
    numSpecies_(nSpecies),
    obsTime_(obsTime),
    movtBan_(obsTime),
    occultsOnlyDC_(occultsOnlyDC),
    maxInfecs_(0),
    numKnownInfecs_(0)
  {
    LoadPopulation(population);
    LoadEpidemic(epidemic);
    SortPopulation();
  }

  Likelihood::Likelihood(const Likelihood& other) : 
    obsTime_(other.obsTime_),
    movtBan_(other.obsTime_),
    occultsOnlyDC_(other.occultsOnlyDC_),
    popSize_(other.popSize_),
    numKnownInfecs_(other.numKnownInfecs_), 
    maxInfecs_(other.maxInfecs_), 
    numSpecies_(other.numSpecies_), 
    population_(other.population_),
    epsilon1_(other.epsilon1_),
    epsilon2_(other.epsilon2_),
    gamma1_(other.gamma1_),
    gamma2_(other.gamma2_),
    delta_(other.delta_),
    omega_(other.omega_),
    nu_(other.nu_),
    alpha_(other.alpha_),
    a_(other.a_),
    b_(other.b_),
    xi_(other.xi_),
    psi_(other.psi_),
    zeta_(other.zeta_),
    phi_(other.phi_)
  {

  }

  const Likelihood&
  Likelihood::operator=(const Likelihood& other)
  {
    // Host Parameters Copy
    epsilon1_ = other.epsilon1_;
    epsilon2_ = other.epsilon2_;
    gamma1_ = other.gamma1_;
    gamma2_ = other.gamma2_;
    delta_ = other.delta_;
    omega_ = other.omega_;
    nu_ = other.nu_;
    alpha_ = other.alpha_;
    a_ = other.a_;
    b_ = other.b_;

    xi_ = other.xi_;
    psi_ = other.psi_;
    zeta_ = other.zeta_;
    phi_ = other.phi_;

    return this->assign(other);
  }

  void
  Likelihood::LoadPopulation(PopDataImporter& importer)
  {
    idMap_.clear();
    population_.clear();

    importer.open();
    try
      {
        size_t idx = 0;
        while (1)
          {
            PopDataImporter::Record record = importer.next();
            Covars covars;
            covars.id = record.id;
            covars.status = SUSC;
            covars.x = record.data.x;
            covars.y = record.data.y;
            covars.I = obsTime_; //EpiRisk::POSINF;
            covars.N = obsTime_; //EpiRisk::POSINF;
            covars.R = obsTime_; //EpiRisk::POSINF;
            covars.cattle = record.data.cattle;
            covars.pigs = record.data.pigs;
            covars.sheep = record.data.sheep;
            idMap_.insert(make_pair(covars.id, idx));
            idx++;
            population_.push_back(covars);
          }
      }
    catch (EpiRisk::fileEOF& e)
      {
        // Continue -- this is harmless condition
      }
    catch (...)
      {
        importer.close();
        throw;
      }

    importer.close();
    const_cast<size_t &>(popSize_) = population_.size();

    return;
  }

  void
  Likelihood::LoadEpidemic(EpiDataImporter& importer)
  {
    importer.open();
    try
      {
        while (1)
          {
            EpiDataImporter::Record record = importer.next();
            map<string, size_t>::const_iterator map = idMap_.find(record.id);
            if (map == idMap_.end())
              {
                cerr << "idMap size: " << idMap_.size() << endl;
                string msg("Key '" + record.id + "' not found in population data");
                throw range_error(msg.c_str());
              }

            Population::iterator ref = population_.begin() + map->second;
            // Check type
            if (record.data.I == EpiRisk::POSINF)
              ref->status = DC;
            else
              ref->status = IP;

            // Check data integrity
            if (record.data.N > record.data.R)
              {
                cerr << "Individual " << record.id
                    << " has N > R.  Setting N = R\n";
                record.data.N = record.data.R;
              }
            if (record.data.R < record.data.I
                and record.data.I != EpiRisk::POSINF)
              {
                cerr << "WARNING: Individual " << record.id
                    << " has I > R!  Setting I = R-7\n";
                record.data.I = record.data.R - 7;
              }

            ref->I = record.data.I;
            ref->N = record.data.N;
            ref->R = record.data.R;

            ref->R = min(ref->R, obsTime_);
            ref->N = min(ref->N, ref->R);
            ref->I = min(ref->I, ref->N);

            if (ref->status == IP and ref->I == ref->N)
              ref->I = ref->N - 14.0f; // Todo: Get rid of this hacky fix!!

            const_cast<size_t&>(maxInfecs_)++;
          }

      }
    catch (EpiRisk::fileEOF& e)
      {
        ;
      }
    catch (...)
      {
        throw;
      }

    if (!occultsOnlyDC_) const_cast<size_t&>(maxInfecs_) = population_.size();
    importer.close();

  }

  void
  Likelihood::SortPopulation()
  {
    // Sort individuals by disease status (IPs -> DCs -> SUSCs)
    sort(population_.begin(), population_.end(), CompareByStatus());
    Covars cmp;
    cmp.status = DC;
    Population::iterator topOfIPs = lower_bound(population_.begin(),
        population_.end(), cmp, CompareByStatus());
    const_cast<size_t&>(numKnownInfecs_) = topOfIPs - population_.begin();
    sort(population_.begin(), topOfIPs, CompareByI());

    std::cout << "Population size: " << popSize_ << "\n";
    std::cout << "Num infecs: " << numKnownInfecs_ << "\n";
    std::cout << "Max infecs: " << maxInfecs_ << "\n";

    // Rebuild population ID index
    idMap_.clear();
    Population::const_iterator it = population_.begin();
    for (size_t i = 0; i < population_.size(); i++)
      {
        idMap_.insert(make_pair(it->id, i));
        it++;
      }

  }

  // fp_t
  // Likelihood::LoadDistanceMatrix(DistMatrixImporter& importer)
  // {
  //   ublas::mapped_matrix<float>* Dimport = new ublas::mapped_matrix<float>(
  //       maxInfecs_, population_.size());
  //   try
  //     {
  //       importer.open();
  //       while (1)
  //         {
  //           DistMatrixImporter::Record record = importer.next();
  //           map<string, size_t>::const_iterator i = idMap_.find(record.id);
  //           map<string, size_t>::const_iterator j = idMap_.find(record.data.j);
  //           if (i == idMap_.end() or j == idMap_.end())
  //             throw range_error("Key pair not found in population");

  //           if (i != j
  //               and i->second < maxInfecs_ /* Don't require distances with i known susc */)
  //             try
  //               {
  //                 Dimport->operator()(i->second, j->second) =
  //                     record.data.distance * record.data.distance;
  //               }
  //             catch (std::exception& e)
  //               {
  //                 cerr << "Inserting distance |" << i->second << " - "
  //                     << j->second << "| = "
  //                     << record.data.distance * record.data.distance
  //                     << " failed" << endl;
  //                 throw e;
  //               }
  //         }
  //     }
  //   catch (EpiRisk::fileEOF& e)
  //     {
  //       cout << "Imported " << Dimport->nnz() << " distance elements" << endl;
  //     }
  //   catch (exception& e)
  //     {
  //       throw e;
  //     }

  //   // Set up distance matrix
  //   size_t dnnz = Dimport->nnz();
  //   ublas::compressed_matrix<float>* D = new ublas::compressed_matrix<float>(
  //       *Dimport);
  //   int* rowPtr = new int[D->index1_data().size()];
  //   for (size_t i = 0; i < D->index1_data().size(); ++i)
  //     rowPtr[i] = D->index1_data()[i];
  //   int* colInd = new int[D->index2_data().size()];
  //   for (size_t i = 0; i < D->index2_data().size(); ++i)
  //     colInd[i] = D->index2_data()[i];
  //   SetDistance(D->value_data().begin(), rowPtr, colInd);
  //   delete[] rowPtr;
  //   delete[] colInd;
  //   delete D;
  //   delete Dimport;

  //   return dnnz;
  // }


  void
  Likelihood::SetParameters(Parameter& epsilon1, Parameter& epsilon2, Parameter& gamma1,
			    Parameter& gamma2, Parameters& xi, Parameters& psi, Parameters& zeta,
			    Parameters& phi, Parameter& delta, Parameter& omega, Parameter& nu,
			    Parameter& alpha, Parameter& a, Parameter& b)
  {

    epsilon1_ = epsilon1.GetValuePtr();
    epsilon2_ = epsilon2.GetValuePtr();
    gamma1_ = gamma1.GetValuePtr();
    gamma2_ = gamma2.GetValuePtr();
    delta_ = delta.GetValuePtr();
    omega_ = omega.GetValuePtr();
    nu_ = nu.GetValuePtr();
    alpha_ = alpha.GetValuePtr();
    a_ = a.GetValuePtr();
    b_ = b.GetValuePtr();

    xi_.clear();
    psi_.clear();
    zeta_.clear();
    phi_.clear();
    for (size_t p = 0; p < numSpecies_; ++p)
      {
        xi_.push_back(xi[p].GetValuePtr());
        psi_.push_back(psi[p].GetValuePtr());
        zeta_.push_back(zeta[p].GetValuePtr());
        phi_.push_back(phi[p].GetValuePtr());
      }
  }

  void
  Likelihood::SetMovtBan(const float movtBanTime)
  {
    const_cast<fp_t&>(movtBan_) = movtBanTime;
  }

  float
  Likelihood::GetMovtBan() const
  {
	  return movtBan_;
  }

  size_t
  Likelihood::GetNumKnownInfecs() const
  {
    return numKnownInfecs_;
  }

  size_t
  Likelihood::GetMaxInfecs() const
  {
    return maxInfecs_;
  }

  size_t
  Likelihood::GetPopulationSize() const
  {
	  return popSize_;
  }

  void
  Likelihood::GetIds(std::vector<std::string>& ids) const
  {
	  ids.resize(popSize_);
	  for(size_t i=0; i<popSize_; ++i)
		  ids[i] = population_[i].id;
  }

  
} // EpiRisk
