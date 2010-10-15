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


template <class Index>
class STLIndexIterator : public std::iterator<typename std::bidirectional_iterator_tag, typename Index::cont_type::const_iterator>
{
	typename Index::index_type::const_iterator iter_;
	
public:
	STLIndexIterator(const typename Index::index_type::iterator& it) : iter_(it) {}
	STLIndexIterator& operator++() { ++iter_; return *this; }
	STLIndexIterator& operator--() { --iter_; return *this; }
	bool operator==(const STLIndexIterator& rhs) { return iter_==rhs.iter_; }
	bool operator!=(const STLIndexIterator& rhs) { return iter_!=rhs.iter_; }
	typename Index::cont_type::value_type& operator*() { return **iter_; }
	const typename Index::cont_type::value_type* operator->() { return &(**iter_); }
};


/// \brief STLIndex indexes an STL container template class
/// 
/// Often it is useful to create a sorted index of elements
/// in an STL container class.  This template class is 
/// designed just to do that.
/// \tparam Container the STL container type to use
/// \tparam Compare a function object comparing the objects \em contained by the container.
template <class Container, class Compare >
class STLIndex
{

	struct Compare_priv
	{
		bool operator() (const typename Container::const_iterator& lhs, const typename Container::const_iterator& rhs) const
		{ Compare compare; return compare(*lhs,*rhs); }
	};
	
	std::multiset<typename Container::const_iterator, Compare_priv> index_;
	
public:
	typedef Container cont_type; ///< the indexed STL container type.
	typedef std::multiset<typename Container::const_iterator,Compare> index_type; ///< The container type used for the index.
	typedef STLIndexIterator< STLIndex<Container, Compare> > Iterator; ///< iterates over the index.
	
	/// Constructor
	///
	/// Complexity \f$O(n)\f$ where \f$n\f$ is the length of 
	/// \param container the STL container to be indexed.
	STLIndex(Container& container)
	{
		typename Container::iterator iter = container.begin();
		while(iter != container.end())
		{
			index_.insert(iter);
			++iter;
		}
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
		return const_iterator(index_.insert(x));
	}
	
	/// \brief Erases an element
	///
	/// This should only be used to maintain the index.  It is the 
	/// caller's responsibility to ensure index consistency.
	void
	erase(Iterator& x)
	{
		index_.erase(x);
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
	operator[](size_t index) {
		assert(index < index_.size());
		
		typename index_type::const_iterator iter = index_.begin();
		advance(iter,index);
		
		return **iter;
	}



};