/*
 * Helper functions. 
 *
 */
#ifndef __HELPERCONTROL_H
#define __HELPERCONTROL_H

#include <vector>
#include "../../core/lib/math/matrix.cpp"
#include "omp.h"
// #include <fstream>
// #include <iostream>
// #include <sstream>
// #include <iterator> // back_inserter

size_t ReduceRotation(size_t index, size_t size);

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalSumRot(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
		const size_t n, const size_t size );

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultTall(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
		const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
		const size_t rows, const size_t cols );

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultWide(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
		const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
		const size_t rows, const size_t cols );

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultWideEf(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
		const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
		const size_t rows, const size_t cols );

/** 
 * Extract the extended diagonal d of a possibly rectangular matrix.
 * There will be min(rows,cols) possible diagonals with max(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 *
 * @return the resulting diagonal as a matrix object.
 */
template<class templElement>
lbcrypto::Matrix<templElement> extractHybridDiag(lbcrypto::Matrix<templElement> const& M, uint32_t d)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= std::min(rows,cols)) {
        throw std::invalid_argument("There are fewer hybrid diagonals than specified. Maybe you want to use extractDiag?");
    }	
	auto zeroAlloc = [=]() { return templElement(0); };
	lbcrypto::Matrix<templElement> result = lbcrypto::Matrix<templElement>(zeroAlloc, std::max(rows,cols), 1);
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < rows; row++)
			result(row,0) = M(row%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < cols; row++)
			result(row,0) = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
	return result;
}

/** 
 * Extract the extended diagonal d of a possibly rectangular matrix.
 * There will be min(rows,cols) possible diagonals with max(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 * @param &result the reference of the result
 *
 */
template<class templElement>
void extractHybridDiag(lbcrypto::Matrix<templElement> const& M, uint32_t d, lbcrypto::Matrix<templElement>& result)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= std::min(rows,cols)) {
        throw std::invalid_argument("There are fewer hybrid diagonals than specified. Maybe you want to use extractDiag?");
    }	
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < rows; row++)
			result(row,0) = M(row%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < cols; row++)
			result(row,0) = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
}

/** 
 * Extract the extended diagonal d of a possibly rectangular matrix.
 * There will be min(rows,cols) possible diagonals with max(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 *
 * @return the resulting diagonal as a std::vector object.
 */
template<class templElement>
std::vector<templElement> extractHybridDiag2Vec(lbcrypto::Matrix<templElement> const& M, uint32_t d)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= std::min(rows,cols)) {
        throw std::invalid_argument("There are fewer hybrid diagonals than specified. Maybe you want to use extractDiag?");
    }	
	std::vector<templElement> result(std::max(rows,cols));
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < rows; row++)
			result[row] = M(row%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < cols; row++)
			result[row] = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
	return result;
}

/** 
 * Extract the extended diagonal d of a possibly rectangular matrix.
 * There will be min(rows,cols) possible diagonals with max(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 * @param &result the reference of the result
 *
 */
template<class templElement>
void extractHybridDiag2Vec(lbcrypto::Matrix<templElement> const& M, uint32_t d, std::vector<templElement>& result)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= std::min(rows,cols)) {
        throw std::invalid_argument("There are fewer hybrid diagonals than specified. Maybe you want to use extractDiag?");
    }	
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < rows; row++)
			result[row] = M(row%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < cols; row++)
			result[row] = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
}


/** 
 * Extract the reduced diagonal d of a possibly rectangular matrix.
 * There will be max(rows,cols) possible diagonals with min(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 *
 * @return the resulting diagonal as a matrix object.
 */
template<class templElement>
lbcrypto::Matrix<templElement> extractDiag(lbcrypto::Matrix<templElement> const& M, uint32_t d)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= std::max(rows,cols)) {
        throw std::invalid_argument("There are fewer diagonals than specified.");
    }	
	auto zeroAlloc = [=]() { return templElement(0); };
	lbcrypto::Matrix<templElement> result = lbcrypto::Matrix<templElement>(zeroAlloc, std::min(rows,cols), 1);
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < cols; row++)
			result(row,0) = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < rows; row++)
			result(row,0) = M(row%rows,(row+d)%cols);
	}
	return result;
}

/** 
 * Extract the reduced diagonal d of a possibly rectangular matrix.
 * There will be max(rows,cols) possible diagonals with min(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 * @param &result the reference of the result
 */
template<class templElement>
void extractDiag(lbcrypto::Matrix<templElement> const& M, uint32_t d, lbcrypto::Matrix<templElement>& result)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= std::max(rows,cols)) {
        throw std::invalid_argument("There are fewer diagonals than specified.");
    }	
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < cols; row++)
			result(row,0) = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < rows; row++)
			result(row,0) = M(row%rows,(row+d)%cols);
	}
}

/** 
 * Extract the reduced diagonal d of a possibly rectangular matrix.
 * There will be max(rows,cols) possible diagonals with min(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 *
 * @return the resulting diagonal as a std::vector object.
 */
template<class templElement>
std::vector<templElement> extractDiag2Vec(lbcrypto::Matrix<templElement> const& M, uint32_t d)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= std::max(rows,cols)) {
        throw std::invalid_argument("There are fewer diagonals than specified.");
    }	
	std::vector<templElement> result(std::min(rows,cols));
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < cols; row++)
			result[row] = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < rows; row++)
			result[row] = M(row%rows,(row+d)%cols);
	}
	return result;
}

/** 
 * Extract the reduced diagonal d of a possibly rectangular matrix.
 * There will be max(rows,cols) possible diagonals with min(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 * @param &result the reference of the result
 */
template<class templElement>
void extractDiag2Vec(lbcrypto::Matrix<templElement> const& M, uint32_t d, std::vector<templElement> &result)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= std::max(rows,cols)) {
        throw std::invalid_argument("There are fewer diagonals than specified.");
    }	
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < cols; row++)
			result[row] = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < rows; row++)
			result[row] = M(row%rows,(row+d)%cols);
	}
}

/** 
 * Transform a row matrix or a column matrix into an std::vector object.
 *
 * @param &M matrix
 * @param &result the reference of the result
 */
template<class templElement>
void mat2Vec(lbcrypto::Matrix<templElement> const& M, std::vector<templElement>& vec)
{
	if ( M.GetRows()!=1 && M.GetCols()!=1 )
		throw std::invalid_argument("This function is designed for matrices that are one row or one column.");

	if ( M.GetRows() == 1 )
	{
#pragma omp parallel for
		for (size_t col = 0; col < M.GetCols(); col++)
			vec[col] = M(0,col);		
	}
	else if ( M.GetCols() == 1 )
	{
#pragma omp parallel for
		for (size_t row = 0; row < M.GetRows(); row++)
			vec[row] = M(row,0);
	}		
}

/** 
 * Transform a row matrix or a column matrix into an std::vector object.
 *
 * @param &M matrix
 * @param &result the reference of the result
 */
template<class templElement>
std::vector<templElement> mat2Vec(lbcrypto::Matrix<templElement> const& M)
{
	if ( M.GetRows()!=1 && M.GetCols()!=1 )
		throw std::invalid_argument("This function is designed for matrices that are one row or one column.");

	std::vector<templElement> vec;

	if ( M.GetRows() == 1 )
	{
		vec.resize(M.GetCols());
#pragma omp parallel for
		for (size_t col = 0; col < M.GetCols(); col++)
			vec[col] = M(0,col);		
	}
	else if ( M.GetCols() == 1 )
	{
		vec.resize(M.GetRows());
#pragma omp parallel for
		for (size_t row = 0; row < M.GetRows(); row++)
			vec[row] = M(row,0);
	}		

	return vec;
}

/** 
 * Transform a matrix into a vector of column vectors.
 *
 * @param &M matrix
 * @param &Cols the reference of the result
 */
template<class templElement>
void mat2Cols(lbcrypto::Matrix<templElement> const& M, std::vector<std::vector<templElement>>& Cols)
{
#pragma omp parallel for
	for (size_t col = 0; col < M.GetCols(); col++){
		mat2Vec(M.ExtractCol(col),Cols[col]);
	}
}

/** 
 * Transform a matrix into a vector of row vectors.
 *
 * @param &M matrix
 * @param &Rows the reference of the result
 */
template<class templElement>
void mat2Rows(lbcrypto::Matrix<templElement> const& M, std::vector<std::vector<templElement>>& Rows)
{
#pragma omp parallel for
	for (size_t row = 0; row < M.GetRows(); row++){
		mat2Vec(M.ExtractRow(row),Rows[row]);
	}
}

/** 
 * Transform a matrix into a vector of vectors that represent the reduced diagonals.
 *
 * @param &M matrix
 * @param &Diags the reference of the result
 */
template<class templElement>
void mat2Diags(lbcrypto::Matrix<templElement> const& M, std::vector<std::vector<templElement>>& Diags)
{
#pragma omp parallel for
	for (size_t i = 0; i < std::max(M.GetRows(),M.GetCols()); i++)
		Diags[i] = extractDiag2Vec(M, i);
}


/** 
 * Transform a matrix into a vector of vectors that represent the extented/hybrid diagonals.
 *
 * @param &M matrix
 * @param &HDiags the reference of the result
 */
template<class templElement>
void mat2HybridDiags(lbcrypto::Matrix<templElement> const& M, std::vector<std::vector<templElement>>& HDiags)
{
#pragma omp parallel for
	for (size_t i = 0; i < std::min(M.GetRows(),M.GetCols()); i++)
		HDiags[i] = extractHybridDiag2Vec(M, i);
}


/**
 * Rotates a vector by an index - left rotation; positive index = left rotation
 *
 * @param &vec input vector.
 * @param index rotation index.
 * @param &result the rotated vector
 */
template<class templElement>
void Rotate(std::vector<templElement> &vec, int32_t index) {

	int32_t size = vec.size();

	std::vector<templElement> copy = vec;

	if (index < 0 || index > size){
		index = ReduceRotation(index,size);
	}

	if (index != 0){
		// two cases: i+index <= slots and i+index > slots
		for(int32_t i = 0; i < size-index; i++)
			vec[i] = copy[i+index];
		for(int32_t i = size-index; i < size; i++)
			vec[i] = copy[i+index-size];
	}
}

template<class templElement>
std::vector<templElement> Fill(const std::vector<templElement> &vec, int slots) {

	int vecSize = vec.size();

	std::vector<templElement> result(slots);

	for (int i = 0; i < slots; i++)
		result[i] = vec[i % vecSize];

	return result;
}


/**
 * Multiplies a wide matrix (rows < cols) by a vector, using short diagonals.
 *
 * @param & M input matrix
 * @param &v input vector.
 * @param &result the resulting vector
 */
template<class templElement>
void matVMultWide(lbcrypto::Matrix<templElement> const& M, std::vector<templElement> const& v, std::vector<templElement>& result)
{
	if ( M.GetCols() != v.size())
		throw std::invalid_argument("# of columns of the matrix does not match the size of the vector.");

	size_t rows = M.GetRows();
	size_t cols = M.GetCols();

	// Obtain the reduced diagonal representation from M	
	std::vector<std::vector<templElement>> shortDiags(cols);
#pragma omp parallel for	
	for (size_t i = 0; i < cols; i++)
	 	shortDiags[i] = std::vector<templElement>(rows);

	mat2Diags(M, shortDiags);

	// Obtain the rotated versions of v
	std::vector<std::vector<templElement>> rotV(cols);
#pragma omp parallel for	
	for (size_t i = 0; i < cols; i++)
	{
	 	rotV[i] = v;
	 	Rotate(rotV[i],i);
	 	// Perform element-wise multiplication, with the result stored in shortDiags
		std::transform( shortDiags[i].begin(), shortDiags[i].end(), rotV[i].begin(), shortDiags[i].begin(),std::multiplies<templElement>() );	
	}


	// Perform binary tree addition
	for (int32_t h = 1; h <= std::ceil(std::log2(cols)); h++){
		for (int32_t i = 0; i < std::ceil(cols/pow(2,h)); i++){
			if (i + std::ceil(cols/pow(2,h)) < std::ceil(cols/pow(2,h-1))){
				std::transform( shortDiags[i].begin(), shortDiags[i].end(), shortDiags[i+std::ceil(cols/pow(2,h))].begin(), shortDiags[i].begin(),std::plus<templElement>() );
			}
		}
	}
	result = shortDiags[0];


}

/**
 * Multiplies a tall matrix (cols < rows) by a vector, using extended diagonals.
 *
 * @param & M input matrix
 * @param &v input vector.
 * @param &result the resulting vector
 */
template<class templElement>
void matVMultTall(lbcrypto::Matrix<templElement> const& M, std::vector<templElement> const& v, std::vector<templElement>& result)
{
	if ( M.GetCols() != v.size())
		throw std::invalid_argument("# of columns of the matrix does not match the size of the vector.");

	size_t rows = M.GetRows();
	size_t cols = M.GetCols();

	// Obtain the extended diagonal representation from M	
	std::vector<std::vector<templElement>> extDiags(cols);
#pragma omp parallel for	
	for (size_t i = 0; i < cols; i++)
	 	extDiags[i] = std::vector<templElement>(rows);

	mat2HybridDiags(M, extDiags);

	// Obtain the rotated versions of v, which now should be v repeated until the closest multiple of the number of cols > the number of rows is reached
	std::vector<std::vector<templElement>> rotV(std::ceil((double)rows/cols)*cols);
	std::vector<templElement> extv = v;
	for (size_t i = 0; i < rows/cols ; i++)
		std::copy(v.begin(),v.end(),back_inserter(extv));


// #pragma omp parallel for	
	for (size_t i = 0; i < cols; i++)
	{
	 	rotV[i] = extv;

	 	Rotate(rotV[i],i);

		std::transform( extDiags[i].begin(), extDiags[i].end(), rotV[i].begin(), extDiags[i].begin(),std::multiplies<templElement>() );	 	
	}


	// Perform binary tree addition
	for (int32_t h = 1; h <= std::ceil(std::log2(cols)); h++){
		for (int32_t i = 0; i < std::ceil((double)cols/pow(2,h)); i++){
			if (i + std::ceil((double)cols/pow(2,h)) < std::ceil(cols/pow(2,h-1))){
				std::transform( extDiags[i].begin(), extDiags[i].end(), extDiags[i+std::ceil(double(cols)/pow(2,h))].begin(), extDiags[i].begin(),std::plus<templElement>() );
			}
		}
	}	
	result = extDiags[0];	

}

/**
 * Multiplies a wide matrix (rows < cols) by a vector, using extended diagonals. 
 * This is possible when rows divides cols.
 *
 * @param & M input matrix
 * @param &v input vector.
 * @param &result the resulting vector
 */
template<class templElement>
void matVMultWideEf(lbcrypto::Matrix<templElement> const& M, std::vector<templElement> const& v, std::vector<templElement>& result)
{
	if ( M.GetCols() != v.size())
		throw std::invalid_argument("# of columns of the matrix does not match the size of the vector.");

	size_t rows = M.GetRows();
	size_t cols = M.GetCols();

	// Obtain the extended diagonal representation from M	
	std::vector<std::vector<templElement>> extDiags(rows);
#pragma omp parallel for	
	for (size_t i = 0; i < rows; i++)
	 	extDiags[i] = std::vector<templElement>(cols);

	mat2HybridDiags(M, extDiags);

	std::vector<templElement> temp(cols);


	std::vector<std::vector<templElement>> rotV(cols);

#pragma omp parallel for	
	for (size_t i = 0; i < rows; i++)
	{
	 	rotV[i] = v;
	 	Rotate(rotV[i],i);
	 	// Perform element-wise multiplication, with the result stored in extDiags
		std::transform( extDiags[i].begin(), extDiags[i].end(), rotV[i].begin(), extDiags[i].begin(),std::multiplies<templElement>() );	 	
	}

	// Perform binary tree addition
	for (int32_t h = 1; h <= std::ceil(std::log2(rows)); h++){
		for (int32_t i = 0; i < std::ceil(double(rows)/pow(2,h)); i++){
			if (i + std::ceil(double(rows)/pow(2,h)) < std::ceil(double(rows)/pow(2,h-1)))
				std::transform( extDiags[i].begin(), extDiags[i].end(), extDiags[i+std::ceil(double(rows)/pow(2,h))].begin(), 
					extDiags[i].begin(),std::plus<templElement>() );
		}
	}

	temp = extDiags[0];


	double q = double(cols)/double(rows);
	double logq = std::ceil(std::log2(q));
	double aq = pow(2,logq);

	// Expand the vector by zeros up to the closest power of 2 from #rows*ceil(log(#columns/#rows))
	std::vector<templElement> res(rows*aq);	
	std::copy(temp.begin(), temp.end(), res.begin());		

	// Perform binary tree rotations
	for (int32_t h = 1; h <= logq; h ++){
		std::transform( res.begin(), res.begin()+rows*(aq/pow(2,h)), 
			res.begin()+rows*(aq/pow(2,h)), res.begin(),std::plus<templElement>() );		
	}

	std::copy(res.begin(), res.begin()+rows, result.begin());	
}

/**
 * Multiplies a matrix by a vector.
 *
 * @param & M input matrix
 * @param &v input vector.
 * @param &result the resulting vector
 */
template<class templElement>
void matVMult(lbcrypto::Matrix<templElement> const& M, std::vector<templElement> const& v, std::vector<templElement>& result)
{
	if ( M.GetCols() != v.size())
		throw std::invalid_argument("# of columns of the matrix does not match the size of the vector.");
	if (M.GetRows() <= M.GetCols()) // wide
		if (M.GetCols() % M.GetRows())
			matVMultWide(M, v, result);
		else // less storage, more rotations
			matVMultWideEf(M, v, result);
	else // tall
		matVMultTall(M, v, result);
}

/**
 * Multiplies a matrix by a vector using the column method.
 *
 * @param & M input matrix
 * @param &v input vector.
 * @param &result the resulting vector
 */
template<class templElement>
void matVMultCol(lbcrypto::Matrix<templElement> const& M, std::vector<templElement> const& v, std::vector<templElement>& result)
{
	if ( M.GetCols() != v.size())
		throw std::invalid_argument("# of columns of the matrix does not match the size of the vector.");

	size_t rows = M.GetRows();
	size_t cols = M.GetCols();

	// Extract the columns of the matrix M
	std::vector<std::vector<templElement>> Cols(cols);
	for (size_t i = 0; i < cols; i++)
		Cols[i] = std::vector<templElement>(rows);
	mat2Cols(M, Cols);

#pragma omp parallel for	
	for (size_t i = 0; i < cols; i++)
	{
		for (size_t j = 0; j < rows; j++)
		{
	 	// Perform multiplication between a column and its corresponding element in v, with the result stored in Cols
			Cols[i][j] *= v[i];
		}
	}	

	// Perform binary tree addition
	for (int32_t h = 1; h <= std::ceil(std::log2(cols)); h++){
		for (int32_t i = 0; i < std::ceil(cols/pow(2,h)); i++){
			if (i + std::ceil(cols/pow(2,h)) < std::ceil(cols/pow(2,h-1)))
				std::transform( Cols[i].begin(), Cols[i].end(), Cols[i+std::ceil(cols/pow(2,h))].begin(), Cols[i].begin(),std::plus<templElement>() );
		}
	}
	result = Cols[0];
}

// reads matrix in std::vector<std::vector>>
template<class templElement>
void readMatrix(std::vector<std::vector<templElement>>& matrix, const std::string filename)
{

    std::ifstream in(filename, std::ifstream::in|std::ios::binary);

    if (in) {
        std::string line;
        
        size_t row = 0;
        while (std::getline(in, line)) {
            std::istringstream is(line);
            // matrix[row] should not be initialized for the back_inserter to work
            std::copy(std::istream_iterator<templElement>(is), std::istream_iterator<templElement>(), std::back_inserter(matrix[row]));
            row++;
        }
    } 
    in.close();
}

// reads matrix in lbcrypto::Matrix
template<class templElement>
void readMatrix(lbcrypto::Matrix<templElement>& matrix, const std::string filename)
{

    std::ifstream in(filename, std::ifstream::in|std::ios::binary);

    if (in) {
        std::string line;
        
        size_t row = 0;
        while (std::getline(in, line)) {
            std::istringstream is(line);
            std::vector<templElement> v;
            // matrix[row] should not be initialized for the back_inserter to work
            std::copy(std::istream_iterator<templElement>(is), std::istream_iterator<templElement>(), std::back_inserter(v));
            matrix.SetRow(row, v);
            row++;
        }
    } 
    in.close();
}

// reads vector in std::vector
template<class templElement>
void readVector(std::vector<templElement>& vec, const std::string filename)
{
    std::ifstream in(filename, std::ifstream::in|std::ios::binary);

    if (in) {
        std::string line;
        std::getline(in, line);
        std::istringstream is(line);
        // vec should not be initialized for the back_inserter to work
        std::copy(std::istream_iterator<templElement>(is), std::istream_iterator<templElement>(), std::back_inserter(vec));
    } 
    in.close();
}


// reads vector in lbcrypto::Matrix, as a row if row=0 and as a column otherwise
template<class templElement>
void readVector(lbcrypto::Matrix<templElement>& vec, const std::string filename, int32_t row)
{
    std::ifstream in(filename, std::ifstream::in|std::ios::binary);

    if (in) {
        std::string line;
        std::getline(in, line);
        std::istringstream is(line);
        std::vector<templElement> v;
        // vec should not be initialized for the back_inserter to work
        std::copy(std::istream_iterator<templElement>(is), std::istream_iterator<templElement>(), std::back_inserter(v));
        if (row == 1)
        	vec.SetRow(0, v);
        else
        	vec.SetCol(0, v);
    } 
    in.close();
}

/**
 * Repeat each element in vec for maxSamp times.
 *
 * @param vec the vector that should be processed.
 * @param size size of vector that is processed.
 * @param maxSamp the maximum number of times one element can be repeated.
 * @param noElem how many times to repeat one element from vec.
 * @return the processed vector.
 */
template<class templElement>
std::vector<templElement> RepeatElements(std::vector<templElement> vec, int32_t size, int32_t maxSamp, int32_t noElem=0){

	if (noElem == 0)
		noElem = maxSamp;
	std::vector<templElement> rep_el_vec(size*maxSamp); 
	for (int32_t i = 0; i < size; i ++)
	{
		for (int32_t j = 0; j < noElem; j ++)
			rep_el_vec[i*maxSamp + j] = vec[i]; 
	}

	return rep_el_vec;
}

//////////////////////// Functions for diagonal matrix-vector multiplication and Rotate and Sum //////////////////////// 
	/**
	* EvalMatVMultTall - Computes the product between a tall matrix and a vector using tree additions
	* @param diags - the matrix is represented as a vector of extended diagonals
	* @param v - the vector to multiply with
	* @param &evalKeys - reference to the map of evaluation keys generated by EvalAutomorphismKeyGen.
	* @param rows - the number of rows of the matrix
	* @param cols - the number of columns of the matrix
	* @return a vector containing the product
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultTall(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
			const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t rows, const size_t cols ) {
		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			
		// Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);	

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));				

		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctxt_vRot(cols);
		ctxt_vRot[0] = ctxt_v; 
	
	// Element-wise multiplication of the extended diagonals with the rotated vectors
	ctxt_vRot[0] = cc->EvalMult( ctxt_diags[0], ctxt_vRot[0] );		
#pragma omp parallel for	
		for (size_t i = 1; i < cols; i++){
			ctxt_vRot[i] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, i, m, vPrecomp, evalKeys);
			ctxt_vRot[i] = cc->EvalMult( ctxt_diags[i], ctxt_vRot[i] );		
		}

		// Perform binary tree addition
		for (int32_t h = 1; h <= std::ceil(std::log2(cols)); h++){
			for (int32_t i = 0; i < std::ceil((double)cols/pow(2,h)); i++){
				if (i + std::ceil((double)cols/pow(2,h)) < std::ceil((double)cols/pow(2,h-1))){
					ctxt_vRot[i] = cc->EvalAdd( ctxt_vRot[i], ctxt_vRot[i+std::ceil((double)cols/pow(2,h))] );

				}					
			}
		}
		result = ctxt_vRot[0];

		return result;
	}		


	/**
	* EvalMatVMultWide - Computes the product between a wide matrix and a vector using tree additions
	* @param diags - the matrix is represented as a vector of extended diagonals
	* @param v - the vector to multiply with
	* @param &evalKeys - reference to the map of evaluation keys generated by EvalAutomorphismKeyGen.
	* @param rows - the number of rows of the matrix
	* @param cols - the number of columns of the matrix
	* @return a vector containing the product
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultWide(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
			const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t rows, const size_t cols ) {
		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			
		// Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);	

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));				

		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctxt_vRot(cols);
		ctxt_vRot[0] = ctxt_v; 
	
	// Element-wise multiplication of the extended diagonals with the rotated vectors
	ctxt_vRot[0] = cc->EvalMult( ctxt_diags[0], ctxt_vRot[0] );		
#pragma omp parallel for	
		for (size_t i = 1; i < cols; i++){
			ctxt_vRot[i] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, i, m, vPrecomp, evalKeys);
			ctxt_vRot[i] = cc->EvalMult( ctxt_diags[i], ctxt_vRot[i] );		
		}

		// Perform binary tree addition
		for (int32_t h = 1; h <= std::ceil(std::log2(cols)); h++){
			for (int32_t i = 0; i < std::ceil((double)cols/pow(2,h)); i++){
				if (i + std::ceil((double)cols/pow(2,h)) < std::ceil((double)cols/pow(2,h-1))){
					ctxt_vRot[i] = cc->EvalAdd( ctxt_vRot[i], ctxt_vRot[i+std::ceil((double)cols/pow(2,h))] );

				}					
			}
		}
		result = ctxt_vRot[0];

		return result;
	}		


	/**
	* EvalMatVMultWideEf - Computes the product between a wide matrix (satisfying rows % cols = 0) 
	* and a vector using tree addition. This is more storage efficient than EvalMatWide.
	* @param ctxt_diags - the matrix is represented as a vector of extended diagonals
	* @param ctxt_v - the vector to multiply with
	* @param rows - the number of rows of the matrix
	* @param cols - the number of columns of the matrix
	* @return a vector containing the product
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultWideEf(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
			const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t rows, const size_t cols ) {

		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			

		// // Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);	

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));				

		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctxt_vRot(cols);
		ctxt_vRot[0] = ctxt_v; 

# pragma omp parallel for 
		for (size_t i = 1; i < cols; ++i)
			ctxt_vRot[i] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, i, m, vPrecomp, evalKeys);

	// Element-wise multiplication of the extended diagonals with the rotated vectors
#pragma omp parallel for	
		for (size_t i = 0; i < rows; i++)
				ctxt_vRot[i] = cc->EvalMult( ctxt_diags[i], ctxt_vRot[i] );

		// Perform binary tree addition
		for (int32_t h = 1; h <= std::ceil(std::log2(rows)); h++){
			for (int32_t i = 0; i < std::ceil((double)rows/pow(2,h)); i++){
				if (i + std::ceil((double)rows/pow(2,h)) < std::ceil((double)rows/pow(2,h-1)))
					ctxt_vRot[i] = cc->EvalAdd( ctxt_vRot[i], ctxt_vRot[i+std::ceil((double)rows/pow(2,h))] );
			}
		}
		result = ctxt_vRot[0];


		double logq = std::ceil(std::log2(double(cols)/double(rows)));
		double aq = pow(2,logq);
		// Assumes that the number of zeros trailing in ctxt_vRot[0] are at least rows*aq - cols 
		// (need zeros up to the closest power of 2 from #rows*ceil(log(#columns/#rows)))

		// Perform binary tree rotations
		for (int32_t h = 1; h <= logq; h++){
			result = cc->EvalAdd( result, cc->GetEncryptionAlgorithm()->EvalAtIndex( result, rows*(aq/pow(2,h)), evalKeys ) );
		}

		return result;
	}
		

	/**
	* EvalSumRot - Computes the rotate and sum procedure for a vector using tree addition 
	* @param ctxt_v - the vector to rotate and sum
	* @param evalKeys - reference to the map of evaluation keys generated by EvalAutomorphismKeyGen
	* @param n - the number of times to rotate
	* @param size - size of the vector
	* @return a vector containing the sum
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalSumRotBatch(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t n, const size_t size ) {
		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();	

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));			

		// /* no optimization, make sure the corresponding rotation keys are generated */
		// // Get cyclotomic order
		// uint32_t m = cc->GetCyclotomicOrder();			
		// // Homomorphic fast rotations with precomputations
		// auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);	
		// for (int32_t i = 1; i < n; i ++ ) 
		// { 
		// 	result = cc->EvalAdd( result, cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, i*(int)size, m, vPrecomp, evalKeys));					
		// }

		/* This optimized version with logarithmic number of operations only works for vectors that have trailing zeros! */
		for (int32_t i = std::ceil(std::log2(n))-1; i >= 0; i --)
		{
			result = cc->EvalAdd( result, cc->GetEncryptionAlgorithm()->EvalAtIndex(result, pow(2,i)*(int)size, evalKeys) );			
		}

		return result;
	}	

	/**
	* EvalSumRot - Computes the rotate and sum procedure for a vector using tree addition 
	* @param ctxt_v - the vector to rotate and sum
	* @param evalKeys - reference to the map of evaluation keys generated by EvalAutomorphismKeyGen
	* @param n - the number of times to rotate
	* @return a vector containing the sum
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalSumExact(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t n ) {
		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();	

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));			

		/* no optimization, make sure the corresponding rotation keys are generated */
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			
		// Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);	
		for (int32_t i = 1; i < (int)n; i ++ ) 
		{ 
			result = cc->EvalAdd( result, cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, i, m, vPrecomp, evalKeys));					
		}

		// /* This optimized version with logarithmic number of operations only works for vectors that have trailing zeros! */
		// for (int32_t i = std::ceil(std::log2(n))-1; i >= 0; i --)
		// {
		// 	result = cc->EvalAdd( result, cc->GetEncryptionAlgorithm()->EvalAtIndex(result, pow(2,i)*(int)size, evalKeys) );			
		// }

		return result;
	}		

	/**
	* EvalSumRot - Computes the rotate and sum procedure for a vector using tree addition 
	* @param ctxt_v - the vector to rotate and sum
	* @param evalKeys - reference to the map of evaluation keys generated by EvalAutomorphismKeyGen
	* @param n - the number of times to rotate
	* @param size - size of the vector
	* @return a vector containing the sum
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalSumRot(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t n, const size_t size ) {
		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			
		// Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);	

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));			

		/* 
		 * No optimization, make sure the correct rotation keys are computed before calling this
		 */
		// for (int32_t i = 1; i < n; i ++ ) 
		// { 
		// 	result = cc->EvalAdd( result, cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, -i*(int)size, m, vPrecomp, evalKeys));					
		// }			

		/*
		 * Improve by using tree addition with all fast rotations, make sure the correct rotation keys are computed before calling this
		 */
		// std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctxt_vRot(n);
		// ctxt_vRot[0] = ctxt_v; 					

		// for (size_t i = 1; i < n; i++)
		// 	ctxt_vRot[i] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, -i*(int)size, m, vPrecomp, evalKeys);
		// for (int32_t h = 1; h <= std::ceil(std::log2(n)); h++)
		// {
		// 	for (int32_t i = 0; i < std::ceil((double)n/pow(2,h)); i++)
		// 	{
		// 		if (i + std::ceil((double)n/pow(2,h)) < std::ceil((double)n/pow(2,h-1)))
		// 			ctxt_vRot[i] = cc->EvalAdd( ctxt_vRot[i], ctxt_vRot[i+std::ceil((double)n/pow(2,h))] );			
		// 	}
		// }
		// result = ctxt_vRot[0];

		/* 
		 * Improve by using tree rotate and sum with regular rotations
		 */	
		for (int32_t h = 1; h <= std::floor(std::log2(n)); h++)
		{
			result = cc->EvalAdd( result, cc->GetEncryptionAlgorithm()->EvalAtIndex(result, -(int)pow(2,h-1)*size, evalKeys) );			
		}
		for (int32_t i = pow(2,std::floor(std::log2(n))); i < (int)n; i ++ ) // for tall matrix
		{ 
			result = cc->EvalAdd( result, cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, -i*(int)size, m, vPrecomp, evalKeys));					
		}				

		return result;
	}


//////////////////////// Functions for diagonal matrix-vector multiplication and Rotate and Sum //////////////////////// 


void CompressEvalKeys(std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &ek, size_t level) {

	const shared_ptr<lbcrypto::LPCryptoParametersCKKS<lbcrypto::DCRTPoly>> cryptoParams =
			std::dynamic_pointer_cast<lbcrypto::LPCryptoParametersCKKS<lbcrypto::DCRTPoly>>(ek.begin()->second->GetCryptoParameters());

	if (cryptoParams->GetKeySwitchTechnique() == lbcrypto::BV) {

		std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>>::iterator it;

		for ( it = ek.begin(); it != ek.end(); it++ )
		{
			std::vector<lbcrypto::DCRTPoly> b = it->second->GetBVector();
			std::vector<lbcrypto::DCRTPoly> a = it->second->GetAVector();

			for (size_t k = 0; k < a.size(); k++) {
				a[k].DropLastElements(level);
				b[k].DropLastElements(level);
			}

			it->second->ClearKeys();

			it->second->SetAVector(std::move(a));
			it->second->SetBVector(std::move(b));
		}
	} else if (cryptoParams->GetKeySwitchTechnique() == lbcrypto::HYBRID) {

		size_t curCtxtLevel = ek.begin()->second->GetBVector()[0].GetParams()->GetParams().size() -
				cryptoParams->GetParamsP()->GetParams().size();
		// size_t curCtxtLevel = ek.begin()->second->GetBVector()[0].GetParams()->GetParams().size() -
		// 		cryptoParams->GetAuxElementParams()->GetParams().size();		

		// current number of levels after compression
		uint32_t newLevels = curCtxtLevel - level;

		std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>>::iterator it;

		for ( it = ek.begin(); it != ek.end(); it++ )
		{
			std::vector<lbcrypto::DCRTPoly> b = it->second->GetBVector();
			std::vector<lbcrypto::DCRTPoly> a = it->second->GetAVector();

			for (size_t k = 0; k < a.size(); k++) {

				auto elementsA = a[k].GetAllElements();
				auto elementsB = b[k].GetAllElements();
				for(size_t i = newLevels; i < curCtxtLevel; i++) {
					elementsA.erase(elementsA.begin() + newLevels);
					elementsB.erase(elementsB.begin() + newLevels);
				}

				a[k] = lbcrypto::DCRTPoly(elementsA);
				b[k] = lbcrypto::DCRTPoly(elementsB);

			}

			it->second->ClearKeys();

			it->second->SetAVector(std::move(a));
			it->second->SetBVector(std::move(b));
		}

	} 
	else {
		PALISADE_THROW(lbcrypto::not_available_error, "Compressed evaluation keys are not currently supported for GHS keyswitching.");
	}


}

/**
 * Ensures that the index for rotation is positive and between 1 and the size of the vector.
 *
 * @param index signed rotation amount.
 * @param size size of vector that is rotated.
 * @return the valid index
 */
size_t ReduceRotation(size_t index, size_t size){

	int32_t isize = int32_t(size);

	// if size is a power of 2
	if ((size & (size - 1)) == 0){
		int32_t n = log2(size);
		if (index >= 0)
			return index - ((index >> n) << n);
		else
			return index+isize + ((int32_t(fabs(index)) >> n) << n);
	}
	else
		return (isize+index%isize)%isize;
}


#endif