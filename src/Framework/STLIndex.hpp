/*
 *  STLIndex.hpp
 *  STLIndex
 *
 *  Created by Chris Jewell on 15/10/2010.
 *  Copyright 2010 University of Warwick. All rights reserved.
 *
 */

#include <iterator>
#include <set>
#include <cassert>


template<class Index>
  class STLIndexIterator : public std::iterator<
      typename std::bidirectional_iterator_tag,
      typename Index::cont_type::const_iterator>
  {
    typename Index::base_iterator iter_;

  public:
    STLIndexIterator(const typename Index::base_iterator& it) :
      iter_(it)
    {
    }
    STLIndexIterator()
    {

    }
    STLIndexIterator(const STLIndexIterator<Index>& copy)
    {
      iter_ = copy.iter_;
    }
    STLIndexIterator&
    operator=(const STLIndexIterator<Index>& assign)
    {
      iter_ = assign.iter_;
    }
    STLIndexIterator&
    operator++()
    {
      ++iter_;
      return *this;
    }
    STLIndexIterator&
    operator++(int)
    {
      iter_++;
      return *this;
    }
    STLIndexIterator&
    operator--()
    {
      --iter_;
      return *this;
    }
    STLIndexIterator&
    operator--(int)
    {
      iter_--;
      return *this;
    }
    bool
    operator==(const STLIndexIterator& rhs)
    {
      return iter_ == rhs.iter_;
    }
    bool
    operator!=(const STLIndexIterator& rhs)
    {
      return iter_ != rhs.iter_;
    }
    typename Index::cont_type::iterator
    operator*()
    {
      return *iter_;
    }
    const typename Index::cont_type::value_type*
    operator->()
    {
      return &(**iter_);
    }
    const typename Index::base_iterator
    base_iterator()
    {
      return iter_;
    }
  };




/// \brief STLIndex indexes an STL container template class
/// 
/// Often it is useful to create a sorted index of elements
/// in an STL container class.  This template class is 
/// designed just to do that.
/// \tparam Container the STL container type to use
/// \tparam Compare a function object comparing the objects \em contained by the container.
template<class Container, class Compare>
  class STLIndex
  {

    struct Compare_priv
    {
      bool
      operator()(const typename Container::const_iterator& lhs,
          const typename Container::const_iterator& rhs) const
      {
        Compare compare;
        return compare(*lhs, *rhs);
      }
    };

    typedef std::multiset<typename Container::iterator, Compare_priv>
              IndexType;

    IndexType index_;


  public:
    typedef Container cont_type; ///< the indexed STL container type.
    typedef typename IndexType::iterator base_iterator;
    typedef STLIndexIterator<STLIndex<Container, Compare> > Iterator; ///< A read-only iterator for the index.

    /// Constructor
    ///
    /// Complexity \f$O(n)\f$ where \f$n\f$ is the length of
    /// \param container the STL container to be indexed.
    STLIndex(Container& container)
    {
      typename Container::iterator iter = container.begin();
      while (iter != container.end())
        {
          index_.insert(iter);
          ++iter;
        }
    }

    /// Default constructor.
    ///
    /// The default constructor does not build the index.  Use
    /// this together with STLIndex<class Container, class Compare>::insert()
    /// to build indices manually.
    STLIndex()
    {

    }

    /// Returns an Iterator to the beginning of the index
    /// \return an Iterator to the beginning of the index
    Iterator
    begin() const
    {
      return Iterator(index_.begin());
    }

    /// Returns an Iterator to just past the end of the index
    /// \return an Iterator to just past the end of the index
    Iterator
    end() const
    {
      return Iterator(index_.end());
    }

    /// \brief Inserts an element into the index
    ///
    /// This should only be used to maintain the index. It is the
    /// caller's responsibility to ensure index consistency.
    /// \return an Iterator to the element just added
    Iterator
    insert(typename Container::iterator& x)
    {
      return Iterator(index_.insert(x));
    }

    /// \brief Erases an element
    ///
    /// This should only be used to maintain the index.  It is the
    /// caller's responsibility to ensure index consistency.
    void
    erase(Iterator& x)
    {
      index_.erase(x.base_iterator());
    }


    void
    erase(typename Container::iterator& x)
    {
      Iterator firstOccurence = index_.find(x);
      Compare_priv compare;

      while(firstOccurence.base_iterator() != index_.end()) { // Need a better condition here!
          if(*(firstOccurence.base_iterator()) == x) {
              index_.erase(firstOccurence.base_iterator());
              break;
          }
          else {
              ++firstOccurence;
          }
      }
    }

    /// \return the size of the index
    size_t
    size() const
    {
      return index_.size();
    }

    /// \brief Allows random access to elements
    ///
    /// This method provides random access to indexed elements by
    /// the position of the element in the index.
    ///
    /// This method operates in constant time for containers which
    /// use random access iterators, and \f$O(i)\f$ time for the \f$i\f$th
    /// element for containers that use bidirectional iterators.
    ///
    /// It is really provided for convenience, and should not be
    /// regarded as anything efficient!
    const typename Container::value_type&
    operator[](size_t index) const
    {
      assert(index < index_.size());

      typename IndexType::const_iterator iter = index_.begin();
      advance(iter, index);

      return **iter;
    }

    /// Clear the index
    void
    clear()
    {
      index_.clear();
    }

    /// Returns an Iterator to the upper_bound
    Iterator
    upper_bound(Iterator& x)
    {
      return Iterator(index_.upper_bound(*x));
    }

    /// Returns an Iterator to the lower_bound
    Iterator
    lower_bound(Iterator& x)
    {
      return Iterator(index_.lower_bound(*x));
    }

  };
