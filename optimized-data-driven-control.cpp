/*
 * Code for model-free control with behavioral paradigm. 
 *
 * This is the optimized and updated functionality described in the paper "Data-driven control on encrypted data", 
 by Andreea B. Alexandru, Anastasios Tsiamis and George J. Pappas.
 * Specifically, we reduce the memory from O((S+T)^2) ciphertexts to O((S+T)) ciphertext. 
 * As a result, this also reduces the number of operations (at least in big Oh notation) 
 * and the number of rotation keys.
 *
 * At every time step, we use both the offline precollected input-output trajectory, as well as the 
 * input-output trajectory collected at the previous time steps to determine the online feedback.
 * The offline preprocessing (including a matrix inversion) is considered to be done separately; online, 
 * matrix-vector encrypted multiplications and summations, along with the inverse of rank-one updated
 * matrix are performed.
 *
 */


// Define PROFILE to enable TIC-TOC timing measurements
#define PROFILE

#include "palisade.h"
#include "Plant.h"
#include "helperControl.h"

using namespace lbcrypto;

// Specify IDENTITY_FOR_TYPE(T) in order to be able to construct an identity matrix
IDENTITY_FOR_TYPE(complex<double>)

void OnlineFeedback();
void OfflineFeedback();

int main()
{
	/* 
	 * Start collecting online measurements to compute the feedback after N steps; 
	 * refresh at every multiple of Trefresh; then, after Tstop steps, do not collect 
	 * new samples anymore. Uses APPROXRESCALE.
	 * Ask the client to compute 1/mSchur immediately after the server computes mSchur. 
	 * This reduces the amount of computation and memory at the cloud (only computes 
	 * the next M_1 once), reduces the noise in u (since the step where the rotated 
	 * mSchur is multiplied to M_1 is not needed anymore) and reduces the depth of u by 1. 
	 * Cut the levels from ciphertexts and keys as needed (after the final refresh).
	 * Add more parallelization.
	 */
	OnlineFbRfSt(); 

	// OfflineFeedback();

	// plantInitRoommm(2,5,5,true);

	return 0;
}

void OnlineFeedback()
{
	TimeVar t0,tt,t1,t2,t3;
	TIC(t0);
	TIC(tt);
	double timeInit(0.0), timeEval(0.0), timeStep(0.0);
	double timeClientmSchur(0.0),timeClientUpdate(0.0), timeClientDec(0.0), timeClientEnc(0.0);
	double timeServer(0.0), timeServerUpdate(0.0), timeServerRefresh(0.0);

	/*
	 * Simulation parameters
	 */
	uint32_t Ton = 8; // Number of online collected samples. Make sure Ton > 0.
	uint32_t Tcont = 5; // Number of time steps to continue the computation after stopping the collection.
	uint32_t Trefresh = 4; // Number of time steps after the server asks for refresh. Make sure 0 < Trefresh.
	// If no refresh is wanted, please set Trefresh = Ton.

	/* 
	 * Initialize the plant
	 */
	uint32_t size = 2;
	Plant<complex<double>>* plant = plantInitRoommm(size, Ton, Tcont); 	

	uint32_t Tstop = plant->N + Ton;	
	uint32_t T = Tstop + Tcont;	

	// Scale M_1 up by a factor = scale (after t >= plant->N)
	std::complex<double> scale = 100;	

//////////////////////// Get inputs: r, yini, uini ////////////////////////
	std::vector<complex<double>> r(plant->getr().GetRows()); // The setpoint
	mat2Vec(plant->getr(), r);
	std::vector<complex<double>> yini(plant->getyini().GetRows()); // This is the initial yini; from now on, we compute it encrypted
	mat2Vec(plant->getyini(), yini);
	std::vector<complex<double>> uini(plant->getuini().GetRows()); // This is the initial uini; from now on, we compute it encrypted
	mat2Vec(plant->getuini(), uini);
//////////////////////// Got inputs: r, yini, uini ////////////////////////	

//////////////////////// These are necessary only for the beginning of the algorithm: Kur, Kyini, Kuini //////////////////////// 
	// Which diagonals to extract depend on the relationship between the 
	// # of rows and the # of columns of the matrices

	Matrix<complex<double>> Kur = plant->getKur(); 
	Matrix<complex<double>> Kyini = plant->getKyini(); 
	Matrix<complex<double>> Kuini = plant->getKuini(); 

	size_t colsKur = Kur.GetCols(); size_t rowsKur = Kur.GetRows();
	size_t colsKyini = Kyini.GetCols(); size_t rowsKyini = Kyini.GetRows(); 
	size_t colsKuini = Kuini.GetCols(); size_t rowsKuini = Kuini.GetRows();


	std::vector<std::vector<complex<double>>> dKur;
	if (rowsKur >= colsKur) // tall
	{
		dKur.resize(colsKur);
	#pragma omp parallel for	
		for (size_t i = 0; i < colsKur; i++)
		 	dKur[i] = std::vector<complex<double>>(rowsKur);		

		mat2HybridDiags(Kur, dKur);	

	 }
	 else // wide
	 	if (colsKur % rowsKur == 0) // wideEf
	 	{
			dKur.resize(rowsKur);
	#pragma omp parallel for	
			for (size_t i = 0; i < rowsKur; i ++)
			 	dKur[i] = std::vector<complex<double>>(colsKur);

			mat2HybridDiags(Kur, dKur);	

		 }	
		 else // plain wide
		 {
			dKur.resize(colsKur);
			for (size_t i = 0; i < colsKur; i ++)
			 	dKur[i] = std::vector<complex<double>>(colsKur);		 	

			mat2Diags(Kur, dKur);
					
		 }

	std::vector<std::vector<complex<double>>> dKyini;
	if (rowsKyini >= colsKyini) // tall
	{
		dKyini.resize(colsKyini);
	#pragma omp parallel for	
		for (size_t i = 0; i < colsKyini; i++)
		 	dKyini[i] = std::vector<complex<double>>(rowsKyini);

		mat2HybridDiags(Kyini, dKyini);	

	 }
	 else // wide
	 	if (colsKyini % rowsKyini == 0) // wideEf
	 	{
			dKyini.resize(rowsKyini);
	#pragma omp parallel for	
			for (size_t i = 0; i < rowsKyini; i ++)
			 	dKyini[i] = std::vector<complex<double>>(colsKyini);

			mat2HybridDiags(Kyini, dKyini);	

		 }	
		 else // plain wide
		 {
			dKyini.resize(colsKyini);
			for (size_t i = 0; i < colsKyini; i ++)
			 	dKyini[i] = std::vector<complex<double>>(colsKyini);		 	

			mat2Diags(Kyini, dKyini);
				
		 }

	std::vector<std::vector<complex<double>>> dKuini;
	if (rowsKuini >= colsKuini) // tall
	{
		dKuini.resize(colsKuini);
	#pragma omp parallel for	
		for (size_t i = 0; i < colsKuini; i++)
		 	dKuini[i] = std::vector<complex<double>>(rowsKuini);

		mat2HybridDiags(Kuini, dKuini);	

	 }
	 else // wide
	 	if (colsKuini % rowsKuini == 0) // wideEf
	 	{
			dKuini.resize(rowsKuini);
	#pragma omp parallel for	
			for (size_t i = 0; i < rowsKuini; i ++)
			 	dKuini[i] = std::vector<complex<double>>(colsKuini);

			mat2HybridDiags(Kuini, dKuini);	

		 }	
		 else // plain wide
		 {
			dKuini.resize(colsKuini);
			for (size_t i = 0; i < colsKuini; i ++)
			 	dKuini[i] = std::vector<complex<double>>(colsKuini);		 	

			mat2Diags(Kuini, dKuini);
				
		 }

//////////////////////// These were necessary only for the beginning of the algorithm: Kur, Kyini, Kuini //////////////////////// 	

//////////////////////// Get inputs: M_1, HY, HU ////////////////////////
	Matrix<complex<double>> M_1 = plant->getM_1(); // The initial inverse matrix = Yf'*Q*Yf + Uf'*R*Uf + lamy*Yp'*Yp + lamu*Up'*Up + lamg*I
	Matrix<complex<double>> HY = plant->getHY(); // The matrix of Hankel matrix for output measurements used for past and future prediction
	Matrix<complex<double>> HU = plant->getHU(); // The matrix of Hankel matrix for input measurements used for past and future prediction

	int32_t S = M_1.GetRows();
	int32_t Ty = HY.GetRows();
	int32_t Tu = HU.GetRows(); 

	// Transform M_1, HY and HU into column representation 
	std::vector<std::vector<complex<double>>> cM_1(S);
	std::vector<std::vector<complex<double>>> cHY(S);
	std::vector<std::vector<complex<double>>> cHU(S);	
	for (int32_t i = 0; i < S; i ++ )
	{
		cM_1[i] = std::vector<complex<double>>(S);
		cHY[i] = std::vector<complex<double>>(Ty);
		cHU[i] = std::vector<complex<double>>(Tu);
	}

	mat2Cols(M_1, cM_1); 
	mat2Cols(HY, cHY); 
	mat2Cols(HU, cHU); 

	// Transform Uf into row representation to use it in the computation of u* without adding an extra masking
	Matrix<complex<double>> Uf = HU.ExtractRows(plant->pu, Tu-1);			
	std::vector<std::vector<complex<double>>> rUf(plant->m);		
	for (size_t i = 0; i < plant->m; i ++ )
		rUf[i] = std::vector<complex<double>>(S);
	mat2Rows(Uf.ExtractRows(0,plant->m-1), rUf);

	int32_t maxSamp = S + max(Ton, plant->M); 

	cout << "S = " << S << ", maxSamp = " << maxSamp << endl;

//////////////////////// Got inputs: M_1, HY, HU ////////////////////////

//////////////////////// Get costs: Q, R, lamg, lamy, lamu ////////////////////////
	Costs<complex<double>> costs = plant->getCosts();
	std::vector<complex<double>> lamQ(plant->py, costs.lamy);
	lamQ.insert(std::end(lamQ), std::begin(costs.diagQ), std::end(costs.diagQ));
	std::vector<complex<double>> lamR(plant->pu, costs.lamu);
	lamR.insert(std::end(lamR), std::begin(costs.diagR), std::end(costs.diagR));	

	std::vector<complex<double>> lamQ_sc = lamQ;
	std::vector<complex<double>> lamR_sc = lamR;
	for (size_t i = 0; i < lamQ_sc.size(); i ++)
		lamQ_sc[i] /= scale;
	for (size_t i = 0; i < lamR_sc.size(); i ++)
		lamR_sc[i] /= scale;	

//////////////////////// Get costs: Q, R, lamg, lamy, lamu ////////////////////////		


	// Step 1: Setup CryptoContext

	// A. Specify main parameters
	/* A1) Multiplicative depth:
	 * The CKKS scheme we setup here will work for any computation
	 * that has a multiplicative depth equal to 'multDepth'.
	 * This is the maximum possible depth of a given multiplication,
	 * but not the total number of multiplications supported by the
	 * scheme.
	 *
	 * For example, computation f(x, y) = x^2 + x*y + y^2 + x + y has
	 * a multiplicative depth of 1, but requires a total of 3 multiplications.
	 * On the other hand, computation g(x_i) = x1*x2*x3*x4 can be implemented
	 * either as a computation of multiplicative depth 3 as
	 * g(x_i) = ((x1*x2)*x3)*x4, or as a computation of multiplicative depth 2
	 * as g(x_i) = (x1*x2)*(x3*x4).
	 *
	 * For performance reasons, it's generally preferable to perform operations
	 * in the shortest multiplicative depth possible.
	 */

	/* For the online model-free control, we need a multDepth = 2t + 7 to compute
	 * the control action at time t. In this case, we assume that the client 
	 * transmits uini and yini at each time step.
	 */

	int32_t multDepth;
	if (Tstop < plant->N) // until the trajectories are concatenated, no deep computations are necessary
		multDepth = 2; 
	else
		multDepth = 2*(Trefresh-1) + 6; //2*(Tstop-plant->N-1) + 6;


	cout << " # of time steps = " << T << ", refresh at time = " << Trefresh << "*k, " <<\
	"stop collecting at time = " << Tstop <<", total circuit depth = " << multDepth << endl << endl;


	/* A2) Bit-length of scaling factor.
	 * CKKS works for real numbers, but these numbers are encoded as integers.
	 * For instance, real number m=0.01 is encoded as m'=round(m*D), where D is
	 * a scheme parameter called scaling factor. Suppose D=1000, then m' is 10 (an
	 * integer). Say the result of a computation based on m' is 130, then at
	 * decryption, the scaling factor is removed so the user is presented with
	 * the real number result of 0.13.
	 *
	 * Parameter 'scaleFactorBits' determines the bit-length of the scaling
	 * factor D, but not the scaling factor itself. The latter is implementation
	 * specific, and it may also vary between ciphertexts in certain versions of
	 * CKKS (e.g., in EXACTRESCALE).
	 *
	 * Choosing 'scaleFactorBits' depends on the desired accuracy of the
	 * computation, as well as the remaining parameters like multDepth or security
	 * standard. This is because the remaining parameters determine how much noise
	 * will be incurred during the computation (remember CKKS is an approximate
	 * scheme that incurs small amounts of noise with every operation). The scaling
	 * factor should be large enough to both accommodate this noise and support results
	 * that match the desired accuracy.
	 */
	uint32_t scaleFactorBits = 53;

	/* A3) Number of plaintext slots used in the ciphertext.
	 * CKKS packs multiple plaintext values in each ciphertext.
	 * The maximum number of slots depends on a security parameter called ring
	 * dimension. In this instance, we don't specify the ring dimension directly,
	 * but let the library choose it for us, based on the security level we choose,
	 * the multiplicative depth we want to support, and the scaling factor size.
	 *
	 * Please use method GetRingDimension() to find out the exact ring dimension
	 * being used for these parameters. Give ring dimension N, the maximum batch
	 * size is N/2, because of the way CKKS works.
	 */

	uint32_t batchSize = maxSamp; // what to display and for EvalSum.

	// The ring dimension is 4*slots
	uint32_t slots = 4096; 

	/* A4) Desired security level based on FHE standards.
	 * This parameter can take four values. Three of the possible values correspond
	 * to 128-bit, 192-bit, and 256-bit security, and the fourth value corresponds
	 * to "NotSet", which means that the user is responsible for choosing security
	 * parameters. Naturally, "NotSet" should be used only in non-production
	 * environments, or by experts who understand the security implications of their
	 * choices.
	 *
	 * If a given security level is selected, the library will consult the current
	 * security parameter tables defined by the FHE standards consortium
	 * (https://homomorphicencryption.org/introduction/) to automatically
	 * select the security parameters. Please see "TABLES of RECOMMENDED PARAMETERS"
	 * in  the following reference for more details:
	 * http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf
	 */


	// SecurityLevel securityLevel = HEStd_128_classic;
	SecurityLevel securityLevel = HEStd_NotSet;

	RescalingTechnique rsTech = APPROXRESCALE;//EXACTRESCALE; 
	KeySwitchTechnique ksTech = HYBRID;	

	uint32_t dnum = 0;
	uint32_t maxDepth = 3;
	// This is the size of the first modulus
	uint32_t firstModSize = 60;	
	uint32_t relinWin = 10;
	MODE mode = OPTIMIZED; // Using ternary distribution

	/* 
	 * The following call creates a CKKS crypto context based on the arguments defined above.
	 */
	CryptoContext<DCRTPoly> cc =
			CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
			   multDepth,
			   scaleFactorBits,
			   batchSize,
			   securityLevel,
			   slots*4, // set this to zero when security level = HEStd_128_classic
			   rsTech,
			   ksTech,
			   dnum,
			   maxDepth,
			   firstModSize,
			   relinWin,
			   mode);


	uint32_t RD = cc->GetRingDimension();
	cout << "CKKS scheme is using ring dimension " << RD << endl;
	uint32_t cyclOrder = RD*2;
	cout << "CKKS scheme is using the cyclotomic order " << cyclOrder << endl;
	cout << "scaleFactorBits = " << scaleFactorBits << ", scale = " << scale.real() << endl << endl;


	// Enable the features that you wish to use
	cc->Enable(ENCRYPTION);
	cc->Enable(SHE);
	cc->Enable(LEVELEDSHE);

	// B. Step 2: Key Generation
	/* B1) Generate encryption keys.
	 * These are used for encryption/decryption, as well as in generating different
	 * kinds of keys.
	 */
	auto keys = cc->KeyGen();

	/* B2) Generate the relinearization key
	 * In CKKS, whenever someone multiplies two ciphertexts encrypted with key s,
	 * we get a result with some components that are valid under key s, and
	 * with an additional component that's valid under key s^2.
	 *
	 * In most cases, we want to perform relinearization of the multiplicaiton result,
	 * i.e., we want to transform the s^2 component of the ciphertext so it becomes valid
	 * under original key s. To do so, we need to create what we call a relinearization
	 * key with the following line.
	 */
	cc->EvalMultKeyGen(keys.secretKey);

	/* B3) Generate the rotation keys
	 * CKKS supports rotating the contents of a packed ciphertext, but to do so, we
	 * need to create what we call a rotation key. This is done with the following call,
	 * which takes as input a vector with indices that correspond to the rotation offset
	 * we want to support. Negative indices correspond to right shift and positive to left
	 * shift. Look at the output of this demo for an illustration of this.
	 *
	 * Keep in mind that rotations work on the entire ring dimension, not the specified
	 * batch size. This means that, if ring dimension is 8 and batch size is 4, then an
	 * input (1,2,3,4,0,0,0,0) rotated by 2 will become (3,4,0,0,0,0,1,2) and not
	 * (3,4,1,2,0,0,0,0). Also, as someone can observe in the output of this demo, since
	 * CKKS is approximate, zeros are not exact - they're just very small numbers.
	 */


	/* 
	 * Find rotation indices
	 */

//////////////////////// These are necessary only for the beginning of the algorithm: rotations for Kur, Kyini, Kuini //////////////////////// 		
	int32_t maxNoRot = max(max(r.size(),yini.size()),uini.size());
	std::vector<int> indexVec(maxNoRot-1);
	std::iota (std::begin(indexVec), std::end(indexVec), 1);

//////////////////////// These were necessary only for the beginning of the algorithm: rotations for Kur, Kyini, Kuini //////////////////////// 	

//////////////////////// Rotations for computing new M_1 //////////////////////// 
	for (int32_t i = 0; i < maxSamp; i ++)
	{
		indexVec.push_back(i);	
		for (int32_t j = 0; j < maxSamp; j ++)
			indexVec.push_back(i-j);
	}
	for (int32_t i = S; i < maxSamp; i ++)
	{
		indexVec.push_back(-i);
	}	
//////////////////////// Rotations for computing new M_1 //////////////////////// 

//////////////////////// Rotations for u = U*M_1*Z //////////////////////// 
	for (int32_t i = 0; i < (int)plant->fu; i ++)
	{
		for (int32_t j = S; j <= maxSamp; j ++ )
			indexVec.push_back((int)((plant->pu+i)*maxSamp-j));		
	}
//////////////////////// Rotations for u = U*M_1*Z //////////////////////// 

//////////////////////// Rotations for constructing the last columns of HY and HU //////////////////////// 	
	indexVec.push_back(maxSamp*plant->p); indexVec.push_back(maxSamp*plant->m);
	indexVec.push_back(int(-Ty+plant->p)*maxSamp); indexVec.push_back(int(-Tu+plant->m)*maxSamp);
//////////////////////// Rotations for constructing the last columns of HY and HU //////////////////////// 		

	// remove any duplicate indices to avoid the generation of extra automorphism keys
	sort( indexVec.begin(), indexVec.end() );
	indexVec.erase( std::unique( indexVec.begin(), indexVec.end() ), indexVec.end() );
	//remove automorphisms corresponding to 0
	indexVec.erase(std::remove(indexVec.begin(), indexVec.end(), 0), indexVec.end());	

	auto EvalRotKeys = cc->GetEncryptionAlgorithm()->EvalAtIndexKeyGen(nullptr, keys.secretKey, indexVec);	

//////////////////////// Rotations for refreshing M_1 ////////////////////////	
	indexVec.clear();
	for (int32_t k = 1; k <= int((Ton-1)/Trefresh); k ++)
	{
		for (int32_t i = 0; i < (int)(S + k*Trefresh); i ++) 
		{
			indexVec.push_back( -(int)(i*(S + k*Trefresh)) ); 
			// indexVec.push_back( -(int)(i*maxSamp) ); // this would be better if all rotation keys would be together
		}		
	}

	// remove any duplicate indices to avoid the generation of extra automorphism keys
	sort( indexVec.begin(), indexVec.end() );
	indexVec.erase( std::unique( indexVec.begin(), indexVec.end() ), indexVec.end() );
	//remove automorphisms corresponding to 0
	indexVec.erase(std::remove(indexVec.begin(), indexVec.end(), 0), indexVec.end());	

	// If refreshing is done multiple times, then there should be different keys, as they do not repeat, and could be deleted afterwards
	auto EvalPackKeys = cc->GetEncryptionAlgorithm()->EvalAtIndexKeyGen(nullptr, keys.secretKey, indexVec);		

	if (Trefresh > 1 && Trefresh < Ton)
		CompressEvalKeys(*EvalPackKeys, (2*(Trefresh-1)+5));

	indexVec.clear();
	for (int32_t k = 1; k <= int((Ton-1)/Trefresh); k ++)
	{
		for (int32_t i = 0; i < (int)(S + k*Trefresh); i ++) 
		{
			indexVec.push_back( (int)(i*(S + k*Trefresh)) );
		}		
	}


	// remove any duplicate indices to avoid the generation of extra automorphism keys
	sort( indexVec.begin(), indexVec.end() );
	indexVec.erase( std::unique( indexVec.begin(), indexVec.end() ), indexVec.end() );
	//remove automorphisms corresponding to 0
	indexVec.erase(std::remove(indexVec.begin(), indexVec.end(), 0), indexVec.end());	

	// If refreshing is done multiple times, then there should be different keys, as they do not repeat, and could be deleted afterwards
	auto EvalUnpackKeys = cc->GetEncryptionAlgorithm()->EvalAtIndexKeyGen(nullptr,keys.secretKey, indexVec);		

//////////////////////// Rotations for refreshing M_1 ////////////////////////		

//////////////////////// Rotations for inner products with rotated elements for time steps after Tstop //////////////////////// 
	indexVec.clear();
	// // REDUNDANCY, not all are needed for EvalSumRotBatch - choose one method and compute only the associated rotations
	// for (int32_t i = 0; i < max(Tu,Ty); i ++)
	// {
	// 	indexVec.push_back(i*maxSamp);	
	// }	
	for (int32_t i = std::ceil(std::log2(max(Tu,Ty)))-1; i >= 0; i --)
		indexVec.push_back(pow(2,i)*maxSamp);		

	// remove any duplicate indices to avoid the generation of extra automorphism keys
	sort( indexVec.begin(), indexVec.end() );
	indexVec.erase( std::unique( indexVec.begin(), indexVec.end() ), indexVec.end() );
	//remove automorphisms corresponding to 0
	indexVec.erase(std::remove(indexVec.begin(), indexVec.end(), 0), indexVec.end());	

	auto EvalSumRotKeys = cc->GetEncryptionAlgorithm()->EvalAtIndexKeyGen(nullptr,keys.secretKey, indexVec);		

//////////////////////// Rotations for inner products with rotated elements //////////////////////// 				

//////////////////////// Rotations for constructing the new yini and uini and the last columns of HY and HU //////////////////////// 	
	indexVec.push_back(maxSamp*plant->p); indexVec.push_back(maxSamp*plant->m);
	indexVec.push_back(int(-plant->py+plant->p)*maxSamp); indexVec.push_back(int(-plant->pu+plant->m)*maxSamp);	

	for(int32_t i = 1; i < (int)plant->m; i ++)	// for u after t>=Tstop
		indexVec.push_back(-i*maxSamp);	

	// remove any duplicate indices to avoid the generation of extra automorphism keys
	sort( indexVec.begin(), indexVec.end() );
	indexVec.erase( std::unique( indexVec.begin(), indexVec.end() ), indexVec.end() );
	//remove automorphisms corresponding to 0
	indexVec.erase(std::remove(indexVec.begin(), indexVec.end(), 0), indexVec.end());	

	auto EvalIniRotKeys = cc->GetEncryptionAlgorithm()->EvalAtIndexKeyGen(nullptr,keys.secretKey, indexVec);			
//////////////////////// Rotations for constructing the new yini and uini and the last columns of HY and HU //////////////////////// 	

	
	// Step 3: Encoding and encryption of inputs

	/* 
	 * Encoding as plaintexts
	 */

	// Vectors r, yini and uini need to be repeated in the packed plaintext for the first time step and each element has to be repeated for S+Ton times for the following time steps
	std::vector<std::complex<double>> rep_r(Fill(r,slots));
	Plaintext ptxt_rep_r = cc->MakeCKKSPackedPlaintext(rep_r);

	std::vector<std::complex<double>> zero_r(Ty*maxSamp); 
	for (int32_t i = 0; i < (int)plant->fy; i ++) //// This can be replaced by online rotations if storage is too large
	{
		for (int32_t j = 0; j < maxSamp; j ++)
			zero_r[plant->py*maxSamp + i*maxSamp + j] = r[i]; 
	}
	Plaintext ptxt_r = cc->MakeCKKSPackedPlaintext(zero_r);

	std::vector<std::complex<double>> rep_yini(Fill(yini,slots));	
	Plaintext ptxt_rep_yini = cc->MakeCKKSPackedPlaintext(rep_yini);

	std::vector<std::complex<double>> repSamp_yini(plant->py*maxSamp); 
	for (int32_t i = 0; i < (int)plant->py; i ++)
	{
		for (int32_t j = 0; j < max(S+(int)plant->M,maxSamp); j ++) // each element needs to be repeated as many times as it will remain in yini => S+plant->M
			repSamp_yini[i*maxSamp + j] = yini[i]; 
	}
	Plaintext ptxt_yini = cc->MakeCKKSPackedPlaintext(repSamp_yini);

	std::vector<std::complex<double>> rep_uini(Fill(uini,slots));	
	Plaintext ptxt_rep_uini = cc->MakeCKKSPackedPlaintext(rep_uini);

	std::vector<std::complex<double>> repSamp_uini(plant->pu*maxSamp); 
	for (int32_t i = 0; i < (int)plant->pu; i ++)
	{
		for (int32_t j = 0; j < max(S+(int)plant->M,maxSamp); j ++) // each element needs to be repeated as many times as it will remain in uini => S+plant->M
			repSamp_uini[i*maxSamp + j] = uini[i]; 
	}	
	Plaintext ptxt_uini = cc->MakeCKKSPackedPlaintext(repSamp_uini);

	Plaintext ptxt_u, ptxt_y;

//////////////////////// These are necessary only for the beginning of the algorithm: encryptions for Kur, Kyini, Kuini //////////////////////// 
	std::vector<Plaintext> ptxt_dKur(dKur.size());
	# pragma omp parallel for 
	for (size_t i = 0; i < dKur.size(); ++i)
		ptxt_dKur[i] = cc->MakeCKKSPackedPlaintext(dKur[i]);

	std::vector<Plaintext> ptxt_dKuini(dKuini.size());
	# pragma omp parallel for 
	for (size_t i = 0; i < dKuini.size(); ++i)
		ptxt_dKuini[i] = cc->MakeCKKSPackedPlaintext(dKuini[i]);

	std::vector<Plaintext> ptxt_dKyini(dKyini.size());
	# pragma omp parallel for 
	for (size_t i = 0; i < dKyini.size(); ++i)
		ptxt_dKyini[i] = cc->MakeCKKSPackedPlaintext(dKyini[i]);

//////////////////////// These were necessary only for the beginning of the algorithm: encryptions for Kur, Kyini, Kuini //////////////////////// 

//////////////////////// Construct plaintexts for M_1, HY, HU, Q, R, lambdas //////////////////////// 

	std::vector<Plaintext> ptxt_cHY(S);
	for (int32_t i = 0; i < S; ++i)
	{
		std::vector<std::complex<double>> repSamp_HY(Ty*maxSamp); // repeat entries of each column of HY
		for (int32_t j = 0; j < Ty; j ++)
		{
			for (int32_t k = 0; k < maxSamp; k ++) 
				repSamp_HY[j*maxSamp + k] = cHY[i][j]; 
		}		
		ptxt_cHY[i] = cc->MakeCKKSPackedPlaintext(repSamp_HY);
	}

	std::vector<Plaintext> ptxt_cHU(S);
	for (int32_t i = 0; i < S; ++i)
	{
		std::vector<std::complex<double>> repSamp_HU(Tu*maxSamp); // repeat entries of each column of HU
		for (int32_t j = 0; j < Tu; j ++)
		{
			for (int32_t k = 0; k < maxSamp; k ++) 
				repSamp_HU[j*maxSamp + k] = cHU[i][j]; 
		}
		ptxt_cHU[i] = cc->MakeCKKSPackedPlaintext(repSamp_HU);	
	}

	std::vector<Plaintext> ptxt_M_1(S);
	for (int32_t i = 0; i < S; ++i) 
	{	
		std::vector<std::complex<double>> scM_1 = cM_1[i];
		std::transform(scM_1.begin(), scM_1.end(), scM_1.begin(), [&scale](std::complex<double>& c){return c*scale;});
		ptxt_M_1[i] = cc->MakeCKKSPackedPlaintext(scM_1);	
	}

	std::vector<Plaintext> ptxt_rUf(plant->m);
	for (size_t i = 0; i < plant->m; ++i) 
	{
		ptxt_rUf[i] = cc->MakeCKKSPackedPlaintext(rUf[i]);	
	}		

	std::vector<std::complex<double>> repSamp_lamQ(Ty*maxSamp); // repeat entries of lamQ
	for (int32_t i = 0; i < Ty; i ++)
	{
		for (int32_t j = 0; j < maxSamp; j ++) 
			repSamp_lamQ[i*maxSamp + j] = lamQ[i]; 
	}
	Plaintext ptxt_lamQ = cc->MakeCKKSPackedPlaintext(repSamp_lamQ);

	std::vector<std::complex<double>> repSamp_lamR(Tu*maxSamp); // repeat entries of lamR
	for (int32_t i = 0; i < Tu; i ++)
	{
		for (int32_t j = 0; j < maxSamp; j ++) 
			repSamp_lamR[i*maxSamp + j] = lamR[i]; 
	}	
	Plaintext ptxt_lamR = cc->MakeCKKSPackedPlaintext(repSamp_lamR);

	Plaintext ptxt_lamg = cc->MakeCKKSPackedPlaintext(std::vector<complex<double>>(maxSamp, costs.lamg));

	std::vector<Plaintext> ptxt_1(maxSamp); //// This can be replaced by online rotations if storage is too large
	std::vector<complex<double>> zero_vec(maxSamp,0);
	for (int32_t i = 0; i < maxSamp; i ++)
	{
		std::vector<complex<double>> elem_vec = zero_vec;
		elem_vec[i] = 1;
		ptxt_1[i] = cc->MakeCKKSPackedPlaintext(elem_vec);
	}
	Plaintext ptxt_scale = cc->MakeCKKSPackedPlaintext({scale});

	std::vector<complex<double>> ones(maxSamp,1);
	Plaintext ptxt_1_v = cc->MakeCKKSPackedPlaintext(ones);

	for (int32_t i = 0; i < Ty; i ++) // repeat entries of sclamQ
	{
		for (int32_t j = 0; j < maxSamp; j ++) 
			repSamp_lamQ[i*maxSamp + j] = lamQ_sc[i]; 
	}
	Plaintext ptxt_sclamQ = cc->MakeCKKSPackedPlaintext(repSamp_lamQ); 

	for (int32_t i = 0; i < Tu; i ++)
	{
		for (int32_t j = 0; j < maxSamp; j ++) 
			repSamp_lamR[i*maxSamp + j] = lamR_sc[i]; 
	}		
	Plaintext ptxt_sclamR = cc->MakeCKKSPackedPlaintext(repSamp_lamR); 

//////////////////////// Constructed plaintexts for M_1, HY, HU, Q, R, lambdas //////////////////////// 
	/* 
	 * Encrypt the encoded vectors 
	 */

//////////////////////// The values for the first iterations until trajectory concatenation //////////////////////// 
	auto ctxt_rep_r = cc->Encrypt(keys.publicKey, ptxt_rep_r);
	auto ctxt_rep_yini = cc->Encrypt(keys.publicKey, ptxt_rep_yini);
	auto ctxt_rep_uini = cc->Encrypt(keys.publicKey, ptxt_rep_uini);	

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKur(dKur.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKur.size(); ++i){
		ctxt_dKur[i] = cc->Encrypt(keys.publicKey, ptxt_dKur[i]);
	}

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKyini(dKyini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKyini.size(); ++i){
		ctxt_dKyini[i] = cc->Encrypt(keys.publicKey, ptxt_dKyini[i]);
	}

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKuini(dKuini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKuini.size(); ++i){
		ctxt_dKuini[i] = cc->Encrypt(keys.publicKey, ptxt_dKuini[i]);	
	}
//////////////////////// The values for the first iterations until trajectory concatenation //////////////////////// 			

	auto ctxt_r = cc->Encrypt(keys.publicKey, ptxt_r);
	Ciphertext<DCRTPoly> ctxt_yini, ctxt_uini;	

	std::vector<Ciphertext<DCRTPoly>> ctxt_cHY(S);
# pragma omp parallel for 
	for (int32_t i = 0; i < S; ++ i)
		ctxt_cHY[i] = cc->Encrypt(keys.publicKey, ptxt_cHY[i]);

	std::vector<Ciphertext<DCRTPoly>> ctxt_cHU(S);
# pragma omp parallel for 
	for (int32_t i = 0; i < S; ++ i)
		ctxt_cHU[i] = cc->Encrypt(keys.publicKey, ptxt_cHU[i]);

	std::vector<Ciphertext<DCRTPoly>> ctxt_M_1(S);
# pragma omp parallel for 
	for (int32_t i = 0; i < S; ++ i)
		ctxt_M_1[i] = cc->Encrypt(keys.publicKey, ptxt_M_1[i]);

	std::vector<Ciphertext<DCRTPoly>> ctxt_rUf(plant->m);
# pragma omp parallel for 
	for (size_t i = 0; i < plant->m; ++ i)
		ctxt_rUf[i] = cc->Encrypt(keys.publicKey, ptxt_rUf[i]);				

	Ciphertext<DCRTPoly> ctxt_scale = cc->Encrypt(keys.publicKey, ptxt_scale);		


	timeInit = TOC(tt);
	cout << "Time for offline key generation, encoding and encryption: " << timeInit << " ms" << endl;

	// Step 4: Evaluation

	TIC(tt);

	Ciphertext<DCRTPoly> ctxt_y, ctxt_u;
	Ciphertext<DCRTPoly> ctxt_mSchur, ctxt_mSchur_1, ctxt_v_mSchur_1, ctxt_scaled_mSchur_1;
	std::vector<Ciphertext<DCRTPoly>> ctxt_mVec(S), ctxt_mVec_s(S); 
	Ciphertext<DCRTPoly> ctxt_M_1mVec, ctxt_M_1mVec_s;
	std::vector<Ciphertext<DCRTPoly>> ctxt_mMMm(S);
	Ciphertext<DCRTPoly> ctxt_M_1_packed;

	// This is necessary to start creating the next column in HU and HY
	std::vector<complex<double>> temp_u = mat2Vec(plant->getHU().ExtractCol(S-1));
	std::vector<complex<double>> temp_y = mat2Vec(plant->getHY().ExtractCol(S-1));

	// Update temp_u, temp_y with uini and yini
	for (size_t j = 0; j < plant->fu; j ++)
		temp_u[j] = temp_u[j+plant->pu];
	for (size_t j = plant->fu; j < plant->pu+plant->fu; j ++)
		temp_u[j] = uini[j-plant->fu];

	for (size_t j = 0; j < plant->fy; j ++)
		temp_y[j] = temp_y[j+plant->py];
	for (size_t j = plant->fy; j < plant->py+plant->fy; j ++)
		temp_y[j] = yini[j-plant->fy];				

	std::vector<std::complex<double>> repSamp_tempy(Ty*maxSamp); // repeat entries of temp_y
	for (int32_t i = 0; i < Ty; i ++)
	{
		for (int32_t j = 0; j < maxSamp; j ++) 
			repSamp_tempy[i*maxSamp + j] = temp_y[i]; 
	}	
	Plaintext ptxt_temp_y = cc->MakeCKKSPackedPlaintext(repSamp_tempy);
	Ciphertext<DCRTPoly> ctxt_temp_y = cc->Encrypt(keys.publicKey, ptxt_temp_y);

	std::vector<std::complex<double>> repSamp_tempu(Tu*maxSamp); // repeat entries of temp_u
	for (int32_t i = 0; i < Tu; i ++)
	{
		for (int32_t j = 0; j < maxSamp; j ++) 
			repSamp_tempu[i*maxSamp + j] = temp_u[i]; 
	}	
	Plaintext ptxt_temp_u = cc->MakeCKKSPackedPlaintext(repSamp_tempu);
	Ciphertext<DCRTPoly> ctxt_temp_u = cc->Encrypt(keys.publicKey, ptxt_temp_u);							

	// Start online computations
	for (size_t t = 0; t < T; t ++)
	{
		cout << "t = " << t << endl << endl;

		TIC(t2);

		TIC(t1);

		if (t < plant->N) // until the trajectory concatenation 
		{

			// Matrix-vector multiplication for Kur*r
			Ciphertext<DCRTPoly> result_r;
			if ( rowsKur >= colsKur ) // tall
				result_r = EvalMatVMultTall(ctxt_dKur, ctxt_rep_r, *EvalRotKeys, rowsKur, colsKur);	
			else // wide
				if ( colsKur % rowsKur == 0) // wide Ef
				{
					result_r = EvalMatVMultWideEf(ctxt_dKur, ctxt_rep_r, *EvalRotKeys, rowsKur, colsKur);	
				}
				else // plain wide
					result_r = EvalMatVMultWide(ctxt_dKur, ctxt_rep_r, *EvalRotKeys, rowsKur, colsKur);								

			// Matrix-vector multiplication for Kyini*yini; 
			Ciphertext<DCRTPoly> result_y;
			if ( rowsKyini >= colsKyini ) // tall
				result_y = EvalMatVMultTall(ctxt_dKyini, ctxt_rep_yini, *EvalRotKeys, rowsKyini, colsKyini);	
			else // wide
				if ( colsKyini % rowsKyini == 0) // wide Ef
					result_y = EvalMatVMultWideEf(ctxt_dKyini, ctxt_rep_yini, *EvalRotKeys, rowsKyini, colsKyini);	
				else // plain wide
					result_y = EvalMatVMultWide(ctxt_dKyini, ctxt_rep_yini, *EvalRotKeys, rowsKyini, colsKyini);	

			// Matrix-vector multiplication for Kuini*uini; 
			Ciphertext<DCRTPoly> result_u;
			if ( rowsKuini >= colsKuini ) // tall
				result_u = EvalMatVMultTall(ctxt_dKuini, ctxt_rep_uini, *EvalRotKeys, rowsKuini, colsKuini);	
			else // wide
				if ( colsKuini % rowsKuini == 0) // wide Ef
					result_u = EvalMatVMultWideEf(ctxt_dKuini, ctxt_rep_uini, *EvalRotKeys, rowsKuini, colsKuini);	
				else // plain wide
					result_u = EvalMatVMultWide(ctxt_dKuini, ctxt_rep_uini, *EvalRotKeys, rowsKuini, colsKuini);					

			// Add the components
			ctxt_u = cc->EvalAdd ( result_u, result_y );
			ctxt_u = cc->EvalAdd ( ctxt_u, result_r );
			
		}
		else // t>=plant->N
		{
			if (t < Tstop)
			{
				ctxt_mSchur = EvalSumRotBatch( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[S], ctxt_cHY[S] ), ptxt_lamQ ),\
					cc->EvalMult( cc->EvalMult( ctxt_cHU[S], ctxt_cHU[S] ), ptxt_lamR ) ), *EvalSumRotKeys, max(Tu,Ty), maxSamp );
				ctxt_mSchur = cc->EvalAdd( ctxt_mSchur, ptxt_lamg );		

				ctxt_mSchur = cc->Rescale(ctxt_mSchur); 				

# pragma omp parallel for
				for (int32_t i = 0; i < S; i ++)
				{
					ctxt_mVec[i] = EvalSumRotBatch( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[S], ctxt_cHY[i] ), ptxt_lamQ ),\
						cc->EvalMult( cc->EvalMult( ctxt_cHU[S], ctxt_cHU[i] ), ptxt_lamR  ) ), *EvalSumRotKeys, max(Tu,Ty), maxSamp );			
				}

	# pragma omp parallel for
				for (int32_t i=0; i < S; i++)
				{
					ctxt_mVec[i] = cc->Rescale(cc->Rescale(ctxt_mVec[i])); 	
				}

				// scaled copy
	# pragma omp parallel for
				for (int32_t i = 0; i < S; i ++)
				{
					ctxt_mVec_s[i] = EvalSumRotBatch( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[S], ctxt_cHY[i] ), ptxt_sclamQ ),\
						cc->EvalMult( cc->EvalMult( ctxt_cHU[S], ctxt_cHU[i] ), ptxt_sclamR  ) ), *EvalSumRotKeys, max(Tu,Ty), maxSamp );					
				}			
		
	# pragma omp parallel for
				for (int32_t i=0; i < S; i++)
				{
					ctxt_mVec_s[i] = cc->Rescale(cc->Rescale(ctxt_mVec_s[i])); 			
				}

				std::vector<Ciphertext<DCRTPoly>> ctxt_temp_vec(S+1);

				ctxt_M_1mVec = cc->EvalMult(ctxt_M_1[0], ctxt_mVec[0]);	

# pragma omp parallel for
				for (int32_t i = 1; i < S; i ++)
				{
					ctxt_temp_vec[i] = cc->EvalMult(ctxt_M_1[i], ctxt_mVec[i]);		
				}				

				for (int32_t i = 1; i < S; i ++)
				{
					ctxt_M_1mVec = cc->EvalAdd(ctxt_M_1mVec, ctxt_temp_vec[i]);		
				}	

				ctxt_M_1mVec = cc->Rescale(ctxt_M_1mVec);												

				// scaled copy
				ctxt_M_1mVec_s = cc->EvalMult(ctxt_M_1[0], ctxt_mVec_s[0]);	

# pragma omp parallel for							
				for (int32_t i = 1; i < S; i ++)
				{
					ctxt_temp_vec[i] = cc->EvalMult(ctxt_M_1[i], ctxt_mVec_s[i]);			
				}	

				for (int32_t i = 1; i < S; i ++)
				{
					ctxt_M_1mVec_s = cc->EvalAdd( ctxt_M_1mVec_s, ctxt_temp_vec[i] );			
				}									
					
				ctxt_M_1mVec_s = cc->Rescale(ctxt_M_1mVec_s);												

				// in the first step where online samples are collected, it is better to perform mVec * (M_1 * mVec);
				// afterwards, it is better to reorder M_1 * (mVec * mVec)	

				if (ctxt_M_1[0]->GetElements()[0].GetParams()->GetParams().size() > ctxt_mVec[0]->GetElements()[0].GetParams()->GetParams().size())
				{
					// Keep the smallest depth for precision
					auto M_1mVecPrecomp = cc->GetEncryptionAlgorithm()->EvalFastRotationPrecompute( ctxt_M_1mVec );	

					ctxt_mSchur = cc->EvalSub( ctxt_mSchur, cc->EvalMult( ctxt_mVec_s[0], ctxt_M_1mVec ) );			

# pragma omp parallel for
					for(int32_t i = 1; i < S; i ++)
					{
						ctxt_temp_vec[i] = cc->EvalMult( ctxt_mVec_s[i], cc->GetEncryptionAlgorithm()->EvalFastRotation( ctxt_M_1mVec, i, cyclOrder, M_1mVecPrecomp, *EvalRotKeys ) );		
					}

					for(int32_t i = 1; i < S; i ++)
					{
						ctxt_mSchur = cc->EvalSub( ctxt_mSchur, ctxt_temp_vec[i]);
					}					
							
					ctxt_mSchur = cc->Rescale(ctxt_mSchur);														

				}
				else
				{

					Ciphertext<DCRTPoly> ctxt_mVecv_s = cc->EvalMult(ctxt_mVec_s[0], ptxt_1[0]);

# pragma omp parallel for
					for (int32_t i = 1; i < S; i ++)
					{
						ctxt_temp_vec[i] = cc->EvalMult(ctxt_mVec_s[i], ptxt_1[i]) ;
					}

					for (int32_t i = 1; i < S; i ++)
					{
						ctxt_mVecv_s = cc->EvalAdd( ctxt_mVecv_s, ctxt_temp_vec[i] );
					}					

					ctxt_mVecv_s = cc->Rescale(ctxt_mVecv_s);

					Ciphertext<DCRTPoly> ctxt_tempSum = cc->EvalMult(ctxt_M_1[0], cc->Rescale(cc->EvalMult(ctxt_mVecv_s, ctxt_mVec[0])));		
									

# pragma omp parallel for
					for (int32_t i = 1; i < S; i ++)
					{
						ctxt_temp_vec[i] = cc->EvalMult(ctxt_M_1[i], cc->Rescale(cc->EvalMult(ctxt_mVecv_s, ctxt_mVec[i])));								
					}

					for (int32_t i = 1; i < S; i ++)
					{
						ctxt_tempSum = cc->EvalAdd( ctxt_tempSum, ctxt_temp_vec[i] );								
					}

					ctxt_mSchur = cc->EvalSub( ctxt_mSchur, EvalSumExact(ctxt_tempSum, *EvalRotKeys, S) ); //cc->EvalSum( ctxt_tempSum, S) ); the latter assumes trailing zeros until the closest power of two to S	
					ctxt_mSchur = cc->Rescale(ctxt_mSchur);										

				}	
								
				timeServer = TOC(t1);
				cout << "Time for computing mSchur at the server at step " << t << ": " << timeServer << " ms" << endl;							

				///////////// At the client: compute 1/mSchur and send it back.		
				TIC(t1);

				Plaintext ptxt_Schur;
				complex<double> mSchur_1;

				cc->Decrypt(keys.secretKey, ctxt_mSchur, &ptxt_Schur);
				ptxt_Schur->SetLength(1);
				mSchur_1 = double(1)/(ptxt_Schur->GetCKKSPackedValue()[0]);

				
				std::vector<complex<double>> v_mSchur_1 = {mSchur_1};
				ctxt_v_mSchur_1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(RepeatElements(v_mSchur_1, 1, S+1),1,0));
				ctxt_mSchur_1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext({mSchur_1},1,0));

				// Make sure to cut the number of towers at re-encryption

				if ( Trefresh > 1 )
				{
					int32_t tLevRed = 0;
					if ((t >= plant->N) && (t-plant->N+1 > (int)((Tstop-plant->N-1)/Trefresh)*Trefresh ) && (t < Tstop)) // after the last refresh
						tLevRed = multDepth + 1 - (5 + 2*(Tstop-t));
					else
						tLevRed = 2*((t-plant->N)%Trefresh+1);

					ctxt_v_mSchur_1 = cc->LevelReduce(ctxt_v_mSchur_1,nullptr, tLevRed);
					ctxt_mSchur_1 = cc->LevelReduce(ctxt_mSchur_1,nullptr, tLevRed);

				}				

		
				timeClientmSchur = TOC(t1);
				cout << "Time for computing and re-encrypting 1/mSchur at the client at step " << t << ": " << timeClientmSchur << " ms" << endl;	

				//////////// Back to the server: continue computation of u

				TIC(t1);

				// We compute M_1 with fewer levels by multiplying 1/mSchur 			
							
				// divide in separate places by scale and by mSchur
				auto SchurPrecomp = cc->GetEncryptionAlgorithm()->EvalFastRotationPrecompute( ctxt_mSchur_1 );	

				std::vector<Ciphertext<DCRTPoly>> ctxt_w(S);

				ctxt_w[0] = cc->EvalMult( ctxt_M_1[0], cc->Rescale(cc->EvalMult(ctxt_mVec[0], ctxt_mSchur_1)) );

		# pragma omp parallel for
				for (int32_t j = 1; j < S; j ++)
				{	
					ctxt_temp_vec[j] = cc->EvalMult( ctxt_M_1[j], cc->Rescale(cc->EvalMult(ctxt_mVec[j], ctxt_mSchur_1)) );	
				}

				for (int32_t j = 1; j < S; j ++)
				{	
					ctxt_w[0] = cc->EvalAdd( ctxt_w[0], ctxt_temp_vec[j] );	
				}				

				for (int32_t i = 1; i < S; i ++)
				{
					// ctxt_w[i] = cc->EvalMult( ctxt_M_1[0], cc->Rescale(cc->EvalMult(ctxt_mVec[0], ctxt_e_mSchur_1[i])) );
					ctxt_w[i] = cc->EvalMult( ctxt_M_1[0], cc->Rescale(cc->EvalMult(ctxt_mVec[0], \
						cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_mSchur_1, -i, cyclOrder, SchurPrecomp, *EvalRotKeys))) ) ;	
		# pragma omp parallel for
					for (int32_t j = 1; j < S; j ++)
					{
						// ctxt_w[i] = cc->EvalAdd( ctxt_w[i], cc->EvalMult( ctxt_M_1[j], cc->Rescale(cc->EvalMult(ctxt_mVec[j], ctxt_e_mSchur_1[i])) ) );	
						ctxt_temp_vec[j] = cc->EvalMult( ctxt_M_1[j], cc->Rescale(cc->EvalMult(ctxt_mVec[j], \
							cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_mSchur_1, -i, cyclOrder, SchurPrecomp, *EvalRotKeys))) );	
					}
					for (int32_t j = 1; j < S; j ++)
					{
						// ctxt_w[i] = cc->EvalAdd( ctxt_w[i], cc->EvalMult( ctxt_M_1[j], cc->Rescale(cc->EvalMult(ctxt_mVec[j], ctxt_e_mSchur_1[i])) ) );	
						ctxt_w[i] = cc->EvalAdd( ctxt_w[i], ctxt_temp_vec[j] );	
					}					
				}	

				ctxt_mMMm.resize(S);

				auto wPrecomp = cc->GetEncryptionAlgorithm()->EvalFastRotationPrecompute( ctxt_w[0] );									
				ctxt_mMMm[0] = ctxt_w[0];		

		# pragma omp parallel for
				for (int32_t j = 1; j < S; j ++)
				{
					// ctxt_mMMm[0] = cc->EvalAdd( ctxt_mMMm[0], cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_w[0], -j, *EvalRotKeys) );	
					ctxt_temp_vec[j] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_w[0], -j, cyclOrder, wPrecomp, *EvalRotKeys) ;
				}

				for (int32_t j = 1; j < S; j ++)
				{
					// ctxt_mMMm[0] = cc->EvalAdd( ctxt_mMMm[0], cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_w[0], -j, *EvalRotKeys) );	
					ctxt_mMMm[0] = cc->EvalAdd( ctxt_mMMm[0], ctxt_temp_vec[j] );
				}

				ctxt_mMMm[0] = cc->EvalMult( ctxt_M_1mVec_s, cc->Rescale(ctxt_mMMm[0]));	
				ctxt_mMMm[0] = cc->Rescale(ctxt_mMMm[0]);

				for (int32_t i = 1; i < S; i ++)
				{
					wPrecomp = cc->GetEncryptionAlgorithm()->EvalFastRotationPrecompute( ctxt_w[i] );
					ctxt_mMMm[i] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_w[i], i, cyclOrder, wPrecomp, *EvalRotKeys);

		# pragma omp parallel for					
					for (int32_t j = 1; j < S; j ++)
					{
						if (i!=j)
						{
							// ctxt_mMMm[i] = cc->EvalAdd( ctxt_mMMm[i], cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_w[i], i-j, *EvalRotKeys) );	
							ctxt_temp_vec[j] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_w[i], i-j, cyclOrder, wPrecomp, *EvalRotKeys);
						}
						else
							ctxt_temp_vec[j] = ctxt_w[i] ;	
					}

					for (int32_t j = 1; j < S; j ++)
					{
						ctxt_mMMm[i] = cc->EvalAdd( ctxt_mMMm[i], ctxt_temp_vec[j] );
					}

					ctxt_mMMm[i] = cc->EvalMult( ctxt_M_1mVec_s, cc->Rescale(ctxt_mMMm[i]));		
					ctxt_mMMm[i] = cc->Rescale(ctxt_mMMm[i]); 
				}														

				// Resize ctxt_M_1, i.e., add an extra column, and make sure to update the columns to account for an extra row
				ctxt_M_1.resize(S+1);	

				if ( ctxt_scale->GetElements()[0].GetParams()->GetParams().size() < ctxt_M_1mVec->GetElements()[0].GetParams()->GetParams().size() )
					ctxt_scale = cc->Encrypt(keys.publicKey, ptxt_scale);
				ctxt_scale = cc->LevelReduce(ctxt_scale, nullptr, ctxt_M_1mVec->GetLevel() - ctxt_scale->GetLevel());
		
				auto scalePrecomp = cc->GetEncryptionAlgorithm()->EvalFastRotationPrecompute( ctxt_scale );	

				ctxt_M_1[S] = cc->EvalAdd( -ctxt_M_1mVec, cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_scale, (int)(-S), cyclOrder, scalePrecomp, *EvalRotKeys) );

				ctxt_M_1[S] = cc->EvalMult(ctxt_M_1[S], ctxt_v_mSchur_1);			
				ctxt_M_1[S] = cc->Rescale(ctxt_M_1[S]);

				ctxt_M_1[0] = cc->EvalSub( cc->EvalAdd( ctxt_M_1[0], ctxt_mMMm[0]), \
					cc->Rescale(cc->GetEncryptionAlgorithm()->EvalAtIndex(cc->EvalMult(ctxt_M_1mVec, ctxt_mSchur_1), (int)(-S), *EvalRotKeys)));	

	# pragma omp parallel for
				for (int32_t i = 1; i < S; i ++)		
				{
					// ctxt_M_1[i] = cc->EvalSub( cc->EvalAdd( ctxt_M_1[i], ctxt_mMMm[i]), cc->Rescale(cc->GetEncryptionAlgorithm()->EvalAtIndex(cc->EvalMult(ctxt_M_1mVec, ctxt_e_mSchur_1[i]), i-S, *EvalRotKeys)));	
					ctxt_temp_vec[i] = cc->Rescale(cc->GetEncryptionAlgorithm()->EvalAtIndex(cc->EvalMult(ctxt_M_1mVec, \
							cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_mSchur_1, -i, cyclOrder, SchurPrecomp, *EvalRotKeys)), i-S, *EvalRotKeys));
				}		

				for (int32_t i = 1; i < S; i ++)		
				{
					// ctxt_M_1[i] = cc->EvalSub( cc->EvalAdd( ctxt_M_1[i], ctxt_mMMm[i]), cc->Rescale(cc->GetEncryptionAlgorithm()->EvalAtIndex(cc->EvalMult(ctxt_M_1mVec, ctxt_e_mSchur_1[i]), i-S, *EvalRotKeys)));		
					ctxt_M_1[i] = cc->EvalSub( cc->EvalAdd( ctxt_M_1[i], ctxt_mMMm[i]), ctxt_temp_vec[i]);
				}																

				// When u_t reaches the maximum allowed multiplicative depth, the server packs M_1 into a vector (if S^2 < RD/2 and into multiple vectors otherwise) 
				// and sends it to the client, that refreshes a single ciphertexts 
				if ( (t > plant->N - 1 ) && ((t-plant->N+1) % Trefresh == 0) && t != Tstop - 1) // in the last case, we don't need to refresh M_1
				{
					TIC(t3);

					ctxt_M_1_packed = ctxt_M_1[0];

	# pragma omp parallel for					
					for (int32_t i = 1; i < S+1; i ++)
					{
						ctxt_temp_vec[i] = cc->GetEncryptionAlgorithm()->EvalAtIndex( ctxt_M_1[i], -(int)(i*(S+1)), *EvalPackKeys );					
					}
		
					for (int32_t i = 1; i < S+1; i ++)
					{
						ctxt_M_1_packed = cc->EvalAdd(ctxt_M_1_packed, ctxt_temp_vec[i] );					
					}					
					// clear keys if this was the last refresh
					if (t-plant->N+1 == int((Tstop-plant->N-1)/Trefresh)*Trefresh)
					{
						EvalPackKeys->clear();
					}	

					timeServerRefresh = TOC(t3);	
					cout << "Time for packing inv(M) at the server at step " << t << ": " << timeServerRefresh << " ms" << endl;					

				}				

				// Compute u*, with rUf represented as rows

				// Add each element of the last column of Uf (last part of the last column of HU) to the corresponding row in ctxt_rUf
				// In order to send only one ciphertext back to the client, the server doesn't send exactly u*, but the client will
				// obtain u* by summing the plant->m batches of length S from what the server sends.

	# pragma omp parallel for
				for (int32_t i = 0; i < (int)plant->m; i ++)
				{
					std::vector<complex<double>> elem_vec(Tu*maxSamp);
					elem_vec[(plant->pu+i)*maxSamp] = 1;						

					ctxt_temp_vec[i] = cc->Rescale(cc->GetEncryptionAlgorithm()->EvalAtIndex( cc->EvalMult(ctxt_cHU[S], \
						cc->MakeCKKSPackedPlaintext({elem_vec})), (int)((plant->pu+i)*maxSamp-S), *EvalRotKeys )) ;					
				}		

				for (int32_t i = 0; i < (int)plant->m; i ++)
				{					
					ctxt_rUf[i] = cc->EvalAdd( ctxt_rUf[i], ctxt_temp_vec[i] );					
				}


				std::vector<Ciphertext<DCRTPoly>> ctxt_Z(S+1);
	# pragma omp parallel for
				for (int32_t i = 0; i < S + 1; i ++)
				{ 
					ctxt_Z[i] = EvalSumRotBatch( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[i], cc->EvalAdd( ctxt_yini, ctxt_r ) ), ptxt_lamQ ),\
						cc->EvalMult( cc->EvalMult( ctxt_cHU[i], ctxt_uini ), ptxt_lamR ) ), *EvalSumRotKeys, max(Tu,Ty), maxSamp );
				
				}		
						

	# pragma omp parallel for
				for (int32_t i = 0; i < S + 1; i ++)
					ctxt_Z[i] = cc->Rescale(cc->Rescale(ctxt_Z[i]));						

				std::vector<Ciphertext<DCRTPoly>> ctxt_uel(plant->m);
				for (size_t k = 0; k < plant->m; k ++)
				{
					ctxt_uel[k] = cc->EvalMult ( ctxt_M_1[0], cc->Rescale(cc->EvalMult( ctxt_rUf[k], ctxt_Z[0] )) );							

		# pragma omp parallel for
					for (int32_t j = 1; j < S+1; ++ j)
						ctxt_temp_vec[j] = cc->EvalMult ( ctxt_M_1[j], cc->Rescale(cc->EvalMult( ctxt_rUf[k], ctxt_Z[j] ) )) ;		

					for (int32_t j = 1; j < S+1; ++ j)
						ctxt_uel[k] = cc->EvalAdd( ctxt_uel[k], ctxt_temp_vec[j] );											
				}							

				ctxt_u = ctxt_uel[0];		

	# pragma omp parallel for
				for(int32_t i = 1; i < (int)plant->m; i ++)	// in this case, we can only send the relevant elements
				{
					ctxt_temp_vec[i] = cc->GetEncryptionAlgorithm()->EvalAtIndex( ctxt_uel[i], -i*maxSamp, *EvalIniRotKeys); //*EvalRotKeys));
				}		

				for(int32_t i = 1; i < (int)plant->m; i ++)	// in this case, we can only send the relevant elements
				{
					ctxt_u = cc->EvalAdd( ctxt_u, ctxt_temp_vec[i]);
				}																

			}
			else // t >= Tstop
			{
				TIC(t1);						

				if (t == Tstop) // update once the last column of the matrix Uf
				{

					for (int32_t i = 0; i < (int)plant->m; i ++)
					{
						std::vector<complex<double>> elem_vec(Tu*maxSamp);
						elem_vec[(plant->pu+i)*maxSamp] = 1;						

						ctxt_rUf[i] = cc->EvalAdd( ctxt_rUf[i], cc->Rescale(cc->GetEncryptionAlgorithm()->EvalAtIndex( cc->EvalMult(ctxt_cHU[S-1], \
							cc->MakeCKKSPackedPlaintext({elem_vec})), (int)((plant->pu+i)*maxSamp-S), *EvalRotKeys )) );

						// ideally, this should be engineered to happen from before (after the last refresh)
						ctxt_rUf[i] = cc->LevelReduce( ctxt_rUf[i], nullptr, ctxt_rUf[i]->GetElements()[0].GetParams()->GetParams().size() - 3); 			
				
					}				

					EvalRotKeys->clear(); // finished using these keys, now we only need the ones for EvalSumRotBatch	

				}		
		
				std::vector<Ciphertext<DCRTPoly>> ctxt_Z(S);
	# pragma omp parallel for
				for (int32_t i = 0; i < S; i ++)
				{ 
					ctxt_Z[i] = EvalSumRotBatch( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[i], cc->EvalAdd( ctxt_yini, ctxt_r ) ), ptxt_lamQ ),\
						cc->EvalMult( cc->EvalMult( ctxt_cHU[i], ctxt_uini ), ptxt_lamR ) ), *EvalSumRotKeys, max(Tu,Ty), maxSamp );
						
				}					
						

	# pragma omp parallel for
				for (int32_t i = 0; i < S; i ++)
					ctxt_Z[i] = cc->Rescale(cc->Rescale(ctxt_Z[i]));				

				std::vector<Ciphertext<DCRTPoly>> ctxt_temp_vec(S+1);

				std::vector<Ciphertext<DCRTPoly>> ctxt_uel(plant->m);
				for (size_t k = 0; k < plant->m; k ++)
				{
					ctxt_uel[k] = cc->EvalMult ( ctxt_M_1[0], cc->Rescale(cc->EvalMult( ctxt_rUf[k], ctxt_Z[0] )) );		

	# pragma omp parallel for
					for (int32_t j = 1; j < S; ++ j)
						ctxt_temp_vec[j] = cc->EvalMult ( ctxt_M_1[j], cc->Rescale(cc->EvalMult( ctxt_rUf[k], ctxt_Z[j] ) ));		

					for (int32_t j = 1; j < S; ++ j)
						ctxt_uel[k] = cc->EvalAdd( ctxt_uel[k], ctxt_temp_vec[j] );											

				}							

				ctxt_u = ctxt_uel[0];		
				for(int32_t i = 1; i < (int)plant->m; i ++)	// in this case, we can only send the relevant elements
				{
					ctxt_u = cc->EvalAdd( ctxt_u, cc->GetEncryptionAlgorithm()->EvalAtIndex( ctxt_uel[i], -i*maxSamp, *EvalIniRotKeys));
				}		
			}



		}
		// cout << "\n# levels of ctxt_u at time " << t << ": " << ctxt_u->GetLevel() << ", depth: " << ctxt_u->GetDepth() <<\
		// ", # towers: " << ctxt_u->GetElements()[0].GetParams()->GetParams().size() << endl << endl;		

		timeServer = TOC(t1);
		cout << "Time for computing the control action at the server at step " << t << ": " << timeServer << " ms" << endl;	

		TIC(t1);

		Plaintext result_u_t;
		cout.precision(8);
		cc->Decrypt(keys.secretKey, ctxt_u, &result_u_t);
		std::vector<complex<double>> u(plant->m);

		if (t < plant->N)
		{
			result_u_t->SetLength(plant->m);
			u = result_u_t->GetCKKSPackedValue(); 

		}
		else
		{
			result_u_t->SetLength(plant->m*maxSamp);
			auto u_pre = result_u_t->GetCKKSPackedValue(); 
			for (size_t i = 0; i < plant->m; i ++)
			{
				if (t < Tstop)
					for (int32_t j = 0; j < S+1; j ++)
						u[i] += u_pre[i*maxSamp + j].real();
				else
					for (int32_t j = 0; j < S; j ++)
						u[i] += u_pre[i*maxSamp + j].real();

			}
		}

		for (size_t i = 0; i < plant->m; i ++)
		{
			u[i].imag(0);		
		}

		if (t >= plant->N)
		{
			for (size_t i = 0; i < plant->m; i ++)
				u[i] /= scale;	
		}

		timeClientDec = TOC(t1);
		cout << "Time for decrypting the control action at the client at step " << t << ": " << timeClientDec << " ms" << endl;	

		TIC(t1);

		// Update plant
		if (t < plant->N)
		{
			plant->updatex(u);
		}
		else 
		{
			plant->onlineUpdatex(u);
			plant->onlineLQR();
		}

		if (plant->M == 1)
		{
			uini = u;
			mat2Vec(plant->gety(),yini);
		}
		else
		{
			Rotate(uini, plant->m);
			std::copy(u.begin(),u.begin()+plant->m,uini.begin()+plant->pu-plant->m);
			Rotate(yini, plant->p);
			std::vector<complex<double>> y(plant->p);
			mat2Vec(plant->gety(),y);
			std::copy(y.begin(),y.begin()+plant->p,yini.begin()+plant->py-plant->p);			
		}


		// plant->printYU(); // if you want to print inputs and outputs at every time step

		plant->setyini(yini);
		plant->setuini(uini);

		timeClientUpdate = TOC(t1);
		cout << "Time for updating the plant at step " << t << ": " << timeClientUpdate << " ms" << endl;		

		if (t < T-1) // we don't need to compute anything else after this
		{
			TIC(t1);

			// Re-encrypt variables 
			// Make sure to cut the number of towers

			if ((t >= plant->N) && (t-plant->N+1 > (int)((Tstop-plant->N-1)/Trefresh)*Trefresh ) && (t < Tstop)) // after the last refresh
			{	
				int32_t tLevRed;
				tLevRed = 5 + 2*(Tstop-t-1); //2*((t-plant->N+1 - (int)((Tstop-plant->N-1)/Trefresh)*Trefresh ));

				ptxt_y = cc->MakeCKKSPackedPlaintext(RepeatElements(mat2Vec(plant->gety()), plant->p, maxSamp),1,0);
				ctxt_y = cc->Encrypt(keys.publicKey, ptxt_y); 		
				ptxt_u = cc->MakeCKKSPackedPlaintext(RepeatElements(u, plant->m, maxSamp),1,0);
				ctxt_u = cc->Encrypt(keys.publicKey, ptxt_u);

				if (Trefresh > 1)
				{
					ctxt_y = cc->LevelReduce(ctxt_y, nullptr, multDepth + 1 - tLevRed);					
					ctxt_u = cc->LevelReduce(ctxt_u, nullptr, multDepth + 1 - tLevRed);		
				}
							
			}
			else
			{
				if (t < plant->N ) // different case because we want the rotated uini and yini - this could be done at the server too, but need more rotations
				{
					ptxt_y = cc->MakeCKKSPackedPlaintext(RepeatElements(mat2Vec(plant->gety()), plant->p, maxSamp));
					ctxt_y = cc->Encrypt(keys.publicKey, ptxt_y);
					ptxt_u = cc->MakeCKKSPackedPlaintext(RepeatElements(u, plant->m, maxSamp));
					ctxt_u = cc->Encrypt(keys.publicKey, ptxt_u);	

					rep_yini = Fill(yini,slots);	
					ptxt_yini = cc->MakeCKKSPackedPlaintext(rep_yini);
					ctxt_rep_yini = cc->Encrypt(keys.publicKey, ptxt_yini);

					rep_uini = Fill(uini,slots);	
					ptxt_uini = cc->MakeCKKSPackedPlaintext(rep_uini);		
					ctxt_rep_uini = cc->Encrypt(keys.publicKey, ptxt_uini);	

					if (t == plant->N - 1)
					{
						ptxt_yini = cc->MakeCKKSPackedPlaintext(RepeatElements(yini, plant->py, maxSamp, S+plant->M));
						ctxt_yini = cc->Encrypt(keys.publicKey, ptxt_yini);
						ptxt_uini = cc->MakeCKKSPackedPlaintext(RepeatElements(uini, plant->pu, maxSamp, S+plant->M));
						ctxt_uini = cc->Encrypt(keys.publicKey, ptxt_uini);		
					
					}					

				}
				else
				{
					if (t < Tstop)
					{
						ptxt_y = cc->MakeCKKSPackedPlaintext(RepeatElements(mat2Vec(plant->gety()), plant->p, maxSamp),1,0);
						ctxt_y = cc->Encrypt(keys.publicKey, ptxt_y); 		
						ptxt_u = cc->MakeCKKSPackedPlaintext(RepeatElements(u, plant->m, maxSamp),1,0);
						ctxt_u = cc->Encrypt(keys.publicKey, ptxt_u);

						if ((t > plant->N - 1) && ((t-plant->N+1)%Trefresh == 0) ) 
						{
							Plaintext result_M_1;
							cc->Decrypt(keys.secretKey, ctxt_M_1_packed, &result_M_1);		
							result_M_1->SetLength((S+1)*(S+1));
							std::vector<complex<double>> M_1_packed((S+1)*(S+1));
							for (size_t i = 0; i < (S+1)*(S+1); i ++)
							{
								M_1_packed[i] = result_M_1->GetCKKSPackedValue()[i];	
							}

							ctxt_M_1_packed = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(M_1_packed)); 						

							if (Trefresh > 1)
							{
								 // the last packing should sometimes have fewer levels than the maximum number
								
								if (t-plant->N+1 == int((Tstop-plant->N-1)/Trefresh)*Trefresh)
								{
									int32_t remLevels = 5 + (Tstop-t-1)*2;							
									ctxt_M_1_packed = cc->LevelReduce(ctxt_M_1_packed, nullptr, multDepth + 1 - remLevels); 						
								}

								if (t-plant->N+1 >= int((Tstop-plant->N-1)/Trefresh)*Trefresh) // after and including the last refresh
								{
									int32_t remLevels = 5 + (Tstop-t-1)*2; // this is how many levels are necessary to compute u* with the minimum # of levels							
									ctxt_y = cc->LevelReduce(ctxt_y, nullptr, multDepth + 1 - remLevels);	
									ctxt_u = cc->LevelReduce(ctxt_u, nullptr, multDepth + 1 - remLevels);	
								}			
							}		


						} 								
					}
					else // t >= Tstop
					{
						int32_t remLevels = 5; // this is how many levels are necessary to compute u* with the minimum # of levels	

						ptxt_y = cc->MakeCKKSPackedPlaintext(RepeatElements(mat2Vec(plant->gety()), plant->p, maxSamp),1,0);
						ctxt_y = cc->Encrypt(keys.publicKey, ptxt_y); 		
						ptxt_u = cc->MakeCKKSPackedPlaintext(RepeatElements(u, plant->m, maxSamp),1,0);
						ctxt_u = cc->Encrypt(keys.publicKey, ptxt_u);
						ctxt_y = cc->LevelReduce(ctxt_y, nullptr, multDepth + 1 - remLevels);	
						ctxt_u = cc->LevelReduce(ctxt_u, nullptr, multDepth + 1 - remLevels);																													
					}	
				}						

			}	

			if (t == Tstop - 1) // client refreshes to keep the computation going until the slots fill up
			{
				ptxt_yini = cc->MakeCKKSPackedPlaintext(RepeatElements(yini, plant->py, maxSamp),1,0);
				ctxt_yini = cc->Encrypt(keys.publicKey, ptxt_yini); 		
				ptxt_uini = cc->MakeCKKSPackedPlaintext(RepeatElements(uini, plant->pu, maxSamp),1,0);
				ctxt_uini = cc->Encrypt(keys.publicKey, ptxt_uini);
				ctxt_yini = cc->LevelReduce(ctxt_yini, nullptr, multDepth + 1 - 5);					
				ctxt_uini = cc->LevelReduce(ctxt_uini, nullptr, multDepth + 1 - 5);				
			
			}					

			timeClientEnc = TOC(t1);
			cout << "Time for encoding and encrypting at the client at time " << t+1 << ": " << timeClientEnc << " ms" << endl;	


			////////////// Back to the server.
			TIC(t1);

			if (t >= plant->N)
			{
				if (t < Tstop)
				{
					if ( ((t-plant->N+1)%Trefresh)== 0 && (t-plant->N+1 <= int((Ton-1)/Trefresh)*Trefresh) )
					{
						if (t < Tstop) // in the case t = Tstop - 1, we don't need to refresh M_1, it should not get here
						{
							TIC(t3);
							// compute digits for fast rotations 
							auto M_1Precomp = cc->GetEncryptionAlgorithm()->EvalFastRotationPrecompute( ctxt_M_1_packed );

							std::vector<complex<double>> ones(S+1,1);
							Plaintext ptxt_1_vec = cc->MakeCKKSPackedPlaintext(ones);
							ctxt_M_1[0] = cc->Rescale( cc->EvalMult( ctxt_M_1_packed, ptxt_1_vec ) );

				# pragma omp parallel for			
							for (int32_t i = 1; i < S+1; i ++)
							{
								// if we want rotation after multiplication, we need to use EvalAtIndex instead of EvalFastRotation
								ctxt_M_1[i] = cc->Rescale( cc->EvalMult( cc->GetEncryptionAlgorithm()->EvalFastRotation(\
											ctxt_M_1_packed, i*(S+1), cyclOrder, M_1Precomp, *EvalUnpackKeys ), ptxt_1_vec ) );
							}					

							timeServerRefresh = TOC(t3);	
							cout << "Time for unpacking inv(M) at the server at step " << t << ": " << timeServerRefresh << " ms" << endl;																
						}						

						// clear keys if this was the last refresh
						if (t-plant->N+1 == int((Tstop-plant->N-1)/Trefresh)*Trefresh)
						{
							EvalUnpackKeys->clear();
						}									
					}	

					S += 1;
				}
			}

			if (t < Tstop)
			{
				if (t == 0 || t >= plant->N)
				{
					ctxt_cHY.resize(S+1); ctxt_cHU.resize(S+1);	
				}

				if (t >= plant->N) 
				{

					if ( (t-plant->N+1 == int((Tstop-plant->N-1)/Trefresh)*Trefresh) && (Trefresh > 1) ) // at the last refresh
					{
						int32_t remLevels = 5 + 2*(Tstop-t-1);
						ctxt_cHY[S-1] = cc->LevelReduce(ctxt_cHY[S-1], nullptr, (int)ctxt_cHY[S-1]->GetElements()[0].GetParams()->GetParams().size()-remLevels);
						ctxt_cHU[S-1] = cc->LevelReduce(ctxt_cHU[S-1], nullptr, (int)ctxt_cHU[S-1]->GetElements()[0].GetParams()->GetParams().size()-remLevels);						
						ctxt_yini = cc->LevelReduce(ctxt_yini, nullptr, (int)ctxt_yini->GetElements()[0].GetParams()->GetParams().size()-remLevels);
						ctxt_uini = cc->LevelReduce(ctxt_uini, nullptr, (int)ctxt_uini->GetElements()[0].GetParams()->GetParams().size()-remLevels);
						ctxt_r = cc->LevelReduce(ctxt_r, nullptr, (int)ctxt_r->GetElements()[0].GetParams()->GetParams().size()-remLevels);
						ptxt_lamQ->SetLevel( remLevels ); //multDepth + 1 - 2*(t-plant->N+1 - (int)((Tstop-plant->N-1)/Trefresh)*Trefresh) );
						ptxt_lamR->SetLevel( remLevels ); //multDepth + 1 - 2*(t-plant->N+1 - (int)((Tstop-plant->N-1)/Trefresh)*Trefresh) );

						CompressEvalKeys(*EvalRotKeys, multDepth + 1 - remLevels);
						CompressEvalKeys(*EvalSumRotKeys, multDepth + 1 - remLevels);
						CompressEvalKeys(*EvalIniRotKeys, multDepth + 1 - remLevels);
						
					}				

					if ( (t-plant->N+1 > (int)((Tstop-plant->N-1)/Trefresh)*Trefresh) && (Trefresh > 1) ) // during and after the last refresh
					{
						ctxt_cHY[S-1] = cc->LevelReduce(ctxt_cHY[S-1], nullptr, 2);
						ctxt_cHU[S-1] = cc->LevelReduce(ctxt_cHU[S-1], nullptr, 2);

						if ( t < Tstop - 1 ) // if t == Tstop - 1, then the client refreshed these values
						{
							ctxt_yini = cc->LevelReduce(ctxt_yini, nullptr, 2);									
							ctxt_uini = cc->LevelReduce(ctxt_uini, nullptr, 2);
						}

						ctxt_r = cc->LevelReduce(ctxt_r, nullptr, 2);
						ptxt_lamQ->SetLevel( 5 + 2*(Tstop-t-1) );
						ptxt_lamR->SetLevel( 5 + 2*(Tstop-t-1) );

						CompressEvalKeys(*EvalRotKeys, 2);
						CompressEvalKeys(*EvalSumRotKeys, 2);
						CompressEvalKeys(*EvalIniRotKeys, 2);							
					}

					ctxt_mVec.resize(S+1); ctxt_mVec_s.resize(S+1);

				}				

				if (t < plant->N) // create new cHY[S] and cHU[S] with the Tini measurements
				{

					if (t == 0)
					{
						ctxt_cHY[S] = ctxt_temp_y;
						ctxt_cHU[S] = ctxt_temp_u;	
					}	

					ctxt_cHY[S] = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_cHY[S], plant->p*maxSamp, *EvalRotKeys),\
						cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_y, int(-Ty+plant->p)*maxSamp, *EvalRotKeys) );
					ctxt_cHU[S] = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_cHU[S], plant->m*maxSamp, *EvalRotKeys),\
						cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_u, int(-Tu+plant->m)*maxSamp, *EvalRotKeys) );	

				}
				else 
				{

					ctxt_cHY[S] = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_cHY[S-1], plant->p*maxSamp, *EvalRotKeys),\
						cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_y, int(-Ty+plant->p)*maxSamp, *EvalRotKeys) );
					ctxt_cHU[S] = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_cHU[S-1], plant->m*maxSamp, *EvalRotKeys),\
						cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_u, int(-Tu+plant->m)*maxSamp, *EvalRotKeys) );								
				}						
				
			}		

			if ( (t >= plant->N) && (t != Tstop - 1) ) // otherwise the client sends the values already encrypted
			{

				if (plant->M == 1)
				{
					ctxt_yini = ctxt_y;
					ctxt_uini = ctxt_u;
				}
				else 
				{

					ctxt_yini = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_yini, plant->p*maxSamp, *EvalIniRotKeys),\
						cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_y, int(-plant->py+plant->p)*maxSamp, *EvalIniRotKeys) );
					ctxt_uini = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_uini, plant->m*maxSamp, *EvalIniRotKeys),\
						cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_u, int(-plant->pu+plant->m)*maxSamp, *EvalIniRotKeys) );										
				}					
			}						

			timeServerUpdate = TOC(t1);
			cout << "Time for updating the Hankel matrices, uini and yini at the server at time " << t+1 << ": " << timeServerUpdate << " ms" << endl;	

			timeStep = TOC(t2);		
			cout << "\nTotal time for evaluation at time " << t << ": " << timeStep << " ms" << endl << endl;	

			cout << "S = " << S << endl;

		}
		else
		{
			timeStep = TOC(t2);		
			cout << "\nTotal time for evaluation at time " << t << ": " << timeStep << " ms" << endl << endl;	
		}					


	}	

	timeEval = TOC(tt);
	cout << "Total time for evaluation for " << T << " steps: " << timeEval << " ms" << endl;	

	timeEval = TOC(t0);
	cout << "Total offline+online time for evaluation for " << T << " steps: " << timeEval << " ms" << endl;	

	plant->printYU(); // print all inputs and outputs at the end of the simulation

	// /* 
	//  * Compare precision to unencrypted results
	//  */

	auto Ym = plant->getYm(); auto Um = plant->getUm(); 
	Plant<complex<double>>* plant_unenc = plantInitRoommm(size, Ton, Tcont, true);


	double max_y = 0;
	double max_u = 0;
	double avg_y = 0;
	double avg_u = 0;
	std::vector<complex<double>> temp(T);

	for (size_t j = 0; j < plant->p; j ++)
	{
		temp = mat2Vec(Ym.ExtractRow(j) - (plant_unenc->getYm()).ExtractRow(j));

		complex<double> temp2;
		for (size_t i = 0; i < T; i++)
		{
			temp2 = std::abs(temp[i]);
			avg_y += temp2.real();
			if (max_y < temp2.real())
				max_y = temp2.real();
		}	
	}

	for (size_t j = 0; j < plant->m; j ++)
	{
		temp = mat2Vec(Um.ExtractRow(j) - (plant_unenc->getUm()).ExtractRow(j));

		complex<double> temp2;
		for (size_t i = 0; i < T; i++)
		{
			temp2 = std::abs(temp[i]);
			avg_u += temp2.real();
			if (max_u < temp2.real())
				max_u = temp2.real();
		}	
	}


	cout << "max diff y: " << max_y << ", avg diff y: " << avg_y/(plant->p*T) << endl;
	cout << "max diff u: " << max_u << ", avg diff u: " << avg_u/(plant->m*T) << endl;	


}


void OfflineFeedback()
{
	TimeVar t,t1,t2;
	TIC(t);
	double timeInit(0.0), timeEval(0.0), timeStep(0.0);
	double timeClientUpdate(0.0), timeClientDec(0.0), timeClientEnc(0.0), timeServer0(0.0), timeServer(0.0);

	/*
	 * Simulation parameters
	 */	
	uint32_t T = 5;

	int32_t flagSendU = 1;	

	uint32_t Trefresh = 3; // if flagSendU = 1, this parameter is automatically set to 1

	/* 
	 * Initialize the plant
	 */
	uint32_t size = 4;	
	Plant<complex<double>>* plant = plantInitRoommm(size, 0, T); 

	// Inputs
	std::vector<complex<double>> r(plant->getr().GetRows());
	mat2Vec(plant->getr(), r);
	std::vector<complex<double>> yini(plant->getyini().GetRows());
	mat2Vec(plant->getyini(), yini);
	std::vector<complex<double>> uini(plant->getuini().GetRows()); 
	mat2Vec(plant->getuini(), uini);

	// Which diagonals to extract depend on the relationship between the 
	// # of rows and the # of columns of the matrices

	Matrix<complex<double>> Kur = plant->getKur(); 
	Matrix<complex<double>> Kyini = plant->getKyini(); 
	Matrix<complex<double>> Kuini = plant->getKuini(); 
	size_t colsKur = Kur.GetCols(); size_t rowsKur = Kur.GetRows();
	size_t colsKyini = Kyini.GetCols(); size_t rowsKyini = Kyini.GetRows(); 
	size_t colsKuini = Kuini.GetCols(); size_t rowsKuini = Kuini.GetRows();


	std::vector<std::vector<complex<double>>> dKur;
	if (rowsKur >= colsKur) // tall
	{
		dKur.resize(colsKur);
	#pragma omp parallel for	
		for (size_t i = 0; i < colsKur; i++)
		 	dKur[i] = std::vector<complex<double>>(rowsKur);

		mat2HybridDiags(Kur, dKur);	
	
	 }
	 else // wide
	 	if (colsKur % rowsKur == 0) // wideEf
	 	{
			dKur.resize(rowsKur);
#pragma omp parallel for	
			for (size_t i = 0; i < rowsKur; i ++)
			 	dKur[i] = std::vector<complex<double>>(colsKur);

			mat2HybridDiags(Kur, dKur);	

		 }	
		 else // plain wide
		 {
			dKur.resize(colsKur);
			for (size_t i = 0; i < colsKur; i ++)
			 	dKur[i] = std::vector<complex<double>>(colsKur);		 	

			mat2Diags(Kur, dKur);
					
		 }

	std::vector<std::vector<complex<double>>> dKyini;
	if (rowsKyini >= colsKyini) // tall
	{
		dKyini.resize(colsKyini);
	#pragma omp parallel for	
		for (size_t i = 0; i < colsKyini; i++)
		 	dKyini[i] = std::vector<complex<double>>(rowsKyini);

		mat2HybridDiags(Kyini, dKyini);	

	 }
	 else // wide
	 	if (colsKyini % rowsKyini == 0) // wideEf
	 	{
			dKyini.resize(rowsKyini);
#pragma omp parallel for	
			for (size_t i = 0; i < rowsKyini; i ++)
			 	dKyini[i] = std::vector<complex<double>>(colsKyini);

			mat2HybridDiags(Kyini, dKyini);	

		 }	
		 else // plain wide
		 {
			dKyini.resize(colsKyini);
			for (size_t i = 0; i < colsKyini; i ++)
			 	dKyini[i] = std::vector<complex<double>>(colsKyini);		 	

			mat2Diags(Kyini, dKyini);
					
		 }

	std::vector<std::vector<complex<double>>> dKuini;
	if (rowsKuini >= colsKuini) // tall
	{
		dKuini.resize(colsKuini);
	#pragma omp parallel for	
		for (size_t i = 0; i < colsKuini; i++)
		 	dKuini[i] = std::vector<complex<double>>(rowsKuini);

		mat2HybridDiags(Kuini, dKuini);	

	 }
	 else // wide
	 	if (colsKuini % rowsKuini == 0) // wideEf
	 	{
			dKuini.resize(rowsKuini);
#pragma omp parallel for	
			for (size_t i = 0; i < rowsKuini; i ++)
			 	dKuini[i] = std::vector<complex<double>>(colsKuini);

			mat2HybridDiags(Kuini, dKuini);	

		 }	
		 else // plain wide
		 {
			dKuini.resize(colsKuini);
			for (size_t i = 0; i < colsKuini; i ++)
			 	dKuini[i] = std::vector<complex<double>>(colsKuini);		 	

			mat2Diags(Kuini, dKuini);
			
		 }

	// Step 1: Setup CryptoContext

	// A. Specify main parameters
	/* A1) Multiplicative depth:
	 * The CKKS scheme we setup here will work for any computation
	 * that has a multiplicative depth equal to 'multDepth'.
	 * This is the maximum possible depth of a given multiplication,
	 * but not the total number of multiplications supported by the
	 * scheme.
	 *
	 * For example, computation f(x, y) = x^2 + x*y + y^2 + x + y has
	 * a multiplicative depth of 1, but requires a total of 3 multiplications.
	 * On the other hand, computation g(x_i) = x1*x2*x3*x4 can be implemented
	 * either as a computation of multiplicative depth 3 as
	 * g(x_i) = ((x1*x2)*x3)*x4, or as a computation of multiplicative depth 2
	 * as g(x_i) = (x1*x2)*(x3*x4).
	 *
	 * For performance reasons, it's generally preferable to perform operations
	 * in the shortest multiplicative depth possible.
	 */
	/* For the one-shot model-free control, we need a multDepth = 2, if the client sends back uini and yini,
	 * and a multDepth = 2t - 1 if we use whatever we want but need to mask and rotate the 
	 * result u* to construct uini.
	*/

	uint32_t multDepth; 
	if (flagSendU == 1)
		multDepth = 2;
	else
		multDepth = 2*Trefresh;

	/* A2) Bit-length of scaling factor.
	 * CKKS works for real numbers, but these numbers are encoded as integers.
	 * For instance, real number m=0.01 is encoded as m'=round(m*D), where D is
	 * a scheme parameter called scaling factor. Suppose D=1000, then m' is 10 (an
	 * integer). Say the result of a computation based on m' is 130, then at
	 * decryption, the scaling factor is removed so the user is presented with
	 * the real number result of 0.13.
	 *
	 * Parameter 'scaleFactorBits' determines the bit-length of the scaling
	 * factor D, but not the scaling factor itself. The latter is implementation
	 * specific, and it may also vary between ciphertexts in certain versions of
	 * CKKS (e.g., in EXACTRESCALE).
	 *
	 * Choosing 'scaleFactorBits' depends on the desired accuracy of the
	 * computation, as well as the remaining parameters like multDepth or security
	 * standard. This is because the remaining parameters determine how much noise
	 * will be incurred during the computation (remember CKKS is an approximate
	 * scheme that incurs small amounts of noise with every operation). The scaling
	 * factor should be large enough to both accommodate this noise and support results
	 * that match the desired accuracy.
	 */
	uint32_t scaleFactorBits = 50;

	/* A3) Number of plaintext slots used in the ciphertext.
	 * CKKS packs multiple plaintext values in each ciphertext.
	 * The maximum number of slots depends on a security parameter called ring
	 * dimension. In this instance, we don't specify the ring dimension directly,
	 * but let the library choose it for us, based on the security level we choose,
	 * the multiplicative depth we want to support, and the scaling factor size.
	 *
	 * Please use method GetRingDimension() to find out the exact ring dimension
	 * being used for these parameters. Give ring dimension N, the maximum batch
	 * size is N/2, because of the way CKKS works.
	 */
	// In the one-shot model-free control, we need the batch size to be the 
	// whole N/2 because we need to pack vectors repeatedly, without trailing 
	// zeros.
	uint32_t batchSize = max(plant->p, plant->m); // what to display
	uint32_t slots = 1024; // has to take into consideration not only security, but dimensions of matrices and how much trailing zeros are neeeded

	/* A4) Desired security level based on FHE standards.
	 * This parameter can take four values. Three of the possible values correspond
	 * to 128-bit, 192-bit, and 256-bit security, and the fourth value corresponds
	 * to "NotSet", which means that the user is responsible for choosing security
	 * parameters. Naturally, "NotSet" should be used only in non-production
	 * environments, or by experts who understand the security implications of their
	 * choices.
	 *
	 * If a given security level is selected, the library will consult the current
	 * security parameter tables defined by the FHE standards consortium
	 * (https://homomorphicencryption.org/introduction/) to automatically
	 * select the security parameters. Please see "TABLES of RECOMMENDED PARAMETERS"
	 * in  the following reference for more details:
	 * http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf
	 */
	// SecurityLevel securityLevel = HEStd_128_classic;
	SecurityLevel securityLevel = HEStd_NotSet;

	RescalingTechnique rsTech = APPROXRESCALE; 
	KeySwitchTechnique ksTech = HYBRID;	

	uint32_t dnum = 0;
	uint32_t maxDepth = 3;
	// This is the size of the first modulus
	uint32_t firstModSize = 60;	
	uint32_t relinWin = 10;
	MODE mode = OPTIMIZED; // Using ternary distribution	

	/* 
	 * The following call creates a CKKS crypto context based on the arguments defined above.
	 */
	CryptoContext<DCRTPoly> cc =
			CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
			   multDepth,
			   scaleFactorBits,
			   batchSize,
			   securityLevel,
			   slots*4, 
			   rsTech,
			   ksTech,
			   dnum,
			   maxDepth,
			   firstModSize,
			   relinWin,
			   mode);

	uint32_t RD = cc->GetRingDimension();
	cout << "CKKS scheme is using ring dimension " << RD << endl;
	uint32_t cyclOrder = RD*2;
	cout << "CKKS scheme is using the cyclotomic order " << cyclOrder << endl << endl;


	// Enable the features that you wish to use
	cc->Enable(ENCRYPTION);
	cc->Enable(SHE);
	cc->Enable(LEVELEDSHE);

	// B. Step 2: Key Generation
	/* B1) Generate encryption keys.
	 * These are used for encryption/decryption, as well as in generating different
	 * kinds of keys.
	 */
	auto keys = cc->KeyGen();

	/* B2) Generate the relinearization key
	 * In CKKS, whenever someone multiplies two ciphertexts encrypted with key s,
	 * we get a result with some components that are valid under key s, and
	 * with an additional component that's valid under key s^2.
	 *
	 * In most cases, we want to perform relinearization of the multiplicaiton result,
	 * i.e., we want to transform the s^2 component of the ciphertext so it becomes valid
	 * under original key s. To do so, we need to create what we call a relinearization
	 * key with the following line.
	 */
	cc->EvalMultKeyGen(keys.secretKey);

	/* B3) Generate the rotation keys
	 * CKKS supports rotating the contents of a packed ciphertext, but to do so, we
	 * need to create what we call a rotation key. This is done with the following call,
	 * which takes as input a vector with indices that correspond to the rotation offset
	 * we want to support. Negative indices correspond to right shift and positive to left
	 * shift. Look at the output of this demo for an illustration of this.
	 *
	 * Keep in mind that rotations work on the entire ring dimension, not the specified
	 * batch size. This means that, if ring dimension is 8 and batch size is 4, then an
	 * input (1,2,3,4,0,0,0,0) rotated by 2 will become (3,4,0,0,0,0,1,2) and not
	 * (3,4,1,2,0,0,0,0). Also, as someone can observe in the output of this demo, since
	 * CKKS is approximate, zeros are not exact - they're just very small numbers.
	 */

	/* 
	 * Find rotation indices
	 */
	size_t maxNoRot = max(max(r.size(),yini.size()),uini.size());
	std::vector<int> indexVec(maxNoRot-1);
	std::iota (std::begin(indexVec), std::end(indexVec), 1);

	if (flagSendU != 1) // to compute new uini at the server
	{
		indexVec.push_back(plant->m);
		indexVec.push_back(-(plant->pu-plant->m));

		// for (size_t i = 1; i < colsKuini; i ++ ) // for less efficient computations
		// 	indexVec.push_back(-i*(int)plant->pu);
		for (size_t i = 1; i <= std::floor(std::log2(colsKuini)); i ++ )
			indexVec.push_back(-pow(2,i-1)*(int)plant->pu);
		for (size_t i = pow(2,std::floor(std::log2(colsKuini))); i < colsKuini; i ++ )
			indexVec.push_back(-i*(int)plant->pu);

	}


	// remove any duplicate indices to avoid the generation of extra automorphism keys
	sort( indexVec.begin(), indexVec.end() );
	indexVec.erase( std::unique( indexVec.begin(), indexVec.end() ), indexVec.end() );
	//remove automorphisms corresponding to 0
	indexVec.erase(std::remove(indexVec.begin(), indexVec.end(), 0), indexVec.end());	

	auto EvalRotKeys = cc->GetEncryptionAlgorithm()->EvalAtIndexKeyGen(nullptr,keys.secretKey, indexVec);	

	/* 
	 * B4) Generate keys for summing up the packed values in a ciphertext
	 */

	// Step 3: Encoding and encryption of inputs

	// Encoding as plaintexts

	// Vectors r, yini and uini need to be repeated in the packed plaintext - bring issues with the rotations in baby step giant step?
	std::vector<std::complex<double>> rep_r(Fill(r,slots));
	Plaintext ptxt_r = cc->MakeCKKSPackedPlaintext(rep_r);

	std::vector<std::complex<double>> rep_yini(Fill(yini,slots));	
	Plaintext ptxt_yini = cc->MakeCKKSPackedPlaintext(rep_yini);

	std::vector<std::complex<double>> rep_uini(Fill(uini,slots));	
	Plaintext ptxt_uini = cc->MakeCKKSPackedPlaintext(rep_uini);

	std::vector<Plaintext> ptxt_dKur(dKur.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKur.size(); ++i)
		ptxt_dKur[i] = cc->MakeCKKSPackedPlaintext(dKur[i]);

	std::vector<Plaintext> ptxt_dKuini(dKuini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKuini.size(); ++i)
		ptxt_dKuini[i] = cc->MakeCKKSPackedPlaintext(dKuini[i]);

	std::vector<Plaintext> ptxt_dKyini(dKyini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKyini.size(); ++i)
		ptxt_dKyini[i] = cc->MakeCKKSPackedPlaintext(dKyini[i]);

	/* 
	 * Encrypt the encoded vectors
	 */
	auto ctxt_r = cc->Encrypt(keys.publicKey, ptxt_r);
	auto ctxt_yini = cc->Encrypt(keys.publicKey, ptxt_yini);
	auto ctxt_uini = cc->Encrypt(keys.publicKey, ptxt_uini);	

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKur(dKur.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKur.size(); ++i)
		ctxt_dKur[i] = cc->Encrypt(keys.publicKey, ptxt_dKur[i]);

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKyini(dKyini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKyini.size(); ++i)
		ctxt_dKyini[i] = cc->Encrypt(keys.publicKey, ptxt_dKyini[i]);

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKuini(dKuini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKuini.size(); ++i)
		ctxt_dKuini[i] = cc->Encrypt(keys.publicKey, ptxt_dKuini[i]);			


	timeInit = TOC(t);
	cout << "Time for offline key generation, encoding and encryption: " << timeInit << " ms" << endl;

	// Step 4: Evaluation

	TIC(t);

	TIC(t1);

	// Kur * r needs to only be performed when r changes! 
	// If the client agrees to send a signal when r changes, then we can save computations.
	// Otherwise, we compute Kur * r at every time step.

	// Matrix-vector multiplication for Kur*r
	Ciphertext<DCRTPoly> result_r;
	if ( rowsKur >= colsKur ) // tall
		result_r = EvalMatVMultTall(ctxt_dKur, ctxt_r, *EvalRotKeys, rowsKur, colsKur);	
	else // wide
		if ( colsKur % rowsKur == 0) // wide Ef
		{
			result_r = EvalMatVMultWideEf(ctxt_dKur, ctxt_r, *EvalRotKeys, rowsKur, colsKur);	
		}
		else // plain wide
			result_r = EvalMatVMultWide(ctxt_dKur, ctxt_r, *EvalRotKeys, rowsKur, colsKur);	

	timeServer0 = TOC(t1);
	cout << "Time for computing the constant values at the server at step 0: " << timeServer0 << " ms" << endl;		


	Ciphertext<DCRTPoly> ctxt_u;

	// Start online computations
	for (size_t t = 0; t < T; t ++)
	{
		cout << "t = " << t << endl << endl;

		TIC(t2);

		TIC(t1);

		// Matrix-vector multiplication for Kyini*yini; 
		Ciphertext<DCRTPoly> result_y;
		if ( rowsKyini >= colsKyini ) // tall
			result_y = EvalMatVMultTall(ctxt_dKyini, ctxt_yini, *EvalRotKeys, rowsKyini, colsKyini);	
		else // wide
			if ( colsKyini % rowsKyini == 0) // wide Ef
			{
				result_y = EvalMatVMultWideEf(ctxt_dKyini, ctxt_yini, *EvalRotKeys, rowsKyini, colsKyini);	
			}
			else // plain wide
				result_y = EvalMatVMultWide(ctxt_dKyini, ctxt_yini, *EvalRotKeys, rowsKyini, colsKyini);	
	
		// Matrix-vector multiplication for Kuini*uini; 
		Ciphertext<DCRTPoly> result_u;
		if ( rowsKuini >= colsKuini ) // tall
			result_u = EvalMatVMultTall(ctxt_dKuini, ctxt_uini, *EvalRotKeys, rowsKuini, colsKuini);	
		else // wide
			if ( colsKuini % rowsKuini == 0) // wide Ef
				result_u = EvalMatVMultWideEf(ctxt_dKuini, ctxt_uini, *EvalRotKeys, rowsKuini, colsKuini);	
			else // plain wide
				result_u = EvalMatVMultWide(ctxt_dKuini, ctxt_uini, *EvalRotKeys, rowsKuini, colsKuini);			

		// Add the components
		ctxt_u = cc->EvalAdd ( result_u, result_y);
		ctxt_u = cc->EvalAdd ( ctxt_u, result_r );			


		timeServer = TOC(t1);
		cout << "Time for computing the control action at the server at step " << t << ": " << timeServer << " ms" << endl;	

		// If the client will not send uini, then the server has to update it
		if (flagSendU != 1 && ( (t+1) % Trefresh != 0 ) ) 
		{
			TIC(t1);

			std::vector<complex<double>> mask(slots);
			for (size_t i = plant->m; i < plant->pu; i ++)
				mask[i] = 1;
			Plaintext mask_ptxt = cc->MakeCKKSPackedPlaintext(mask);
			ctxt_uini = cc->EvalMult(ctxt_uini,mask_ptxt);							

			std::vector<complex<double>> mask_u(slots);
			for (size_t i = 0; i < plant->m; i ++)
				mask_u[i] = 1;		
			mask_ptxt = cc->MakeCKKSPackedPlaintext(mask_u);
			ctxt_u = cc->EvalMult(ctxt_u,mask_ptxt);		

			ctxt_u = cc->Rescale(ctxt_u);		

			if (ReduceRotation(-(plant->pu-plant->m),slots) != 0)
				ctxt_uini = cc->EvalAdd(cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_uini, plant->m, *EvalRotKeys), cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_u, -(plant->pu-plant->m), *EvalRotKeys));	
			else
				ctxt_uini = cc->EvalAdd(cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_uini, plant->m, *EvalRotKeys), ctxt_u);			

			// Only need to repeat uini the amount of times required by the diagonal method in the matrix vector multiplication

			ctxt_uini = EvalSumRot( ctxt_uini, *EvalRotKeys, colsKuini, plant->pu );		

			ctxt_uini = cc->Rescale(ctxt_uini);

			timeServer = TOC(t1);
			cout << "Time for updating uini at the server at time " << t << " for the next time step: " << timeServer << " ms" << endl;		
		}

		TIC(t1);

		Plaintext result_u_t;
		cout.precision(8);
		cc->Decrypt(keys.secretKey, ctxt_u, &result_u_t);
		result_u_t->SetLength(plant->m);

		auto u = result_u_t->GetCKKSPackedValue(); 
		// for (size_t i = 0; i < plant->m; i ++)
			// u[i].imag(0); // Make sure to make the imaginary parts to be zero s.t. error does not accumulate

		timeClientDec = TOC(t1);
		cout << "Time for decrypting the control action at the client at step " << t << ": " << timeClientDec << " ms" << endl;	

		TIC(t1);

		// Update plant
		plant->updatex(u);
		if (plant->M == 1)
		{
			uini = u;
			mat2Vec(plant->gety(),yini);
		}
		else
		{
			Rotate(uini, plant->m);
			std::copy(u.begin(),u.begin()+plant->m,uini.begin()+plant->pu-plant->m);
			Rotate(yini, plant->p);
			std::vector<complex<double>> y(plant->p);
			mat2Vec(plant->gety(),y);
			std::copy(y.begin(),y.begin()+plant->p,yini.begin()+plant->py-plant->p);			
		}

		// plant->printYU(); // if you want to print inputs and outputs at every time step
		plant->setyini(yini);
		plant->setuini(uini);


		timeClientUpdate = TOC(t1);
		cout << "Time for updating the plant at step " << t << ": " << timeClientUpdate << " ms" << endl;		

		TIC(t1);

		// Re-encrypt variables 
		rep_yini = Fill(yini,slots);	
		ptxt_yini = cc->MakeCKKSPackedPlaintext(rep_yini);
		ctxt_yini = cc->Encrypt(keys.publicKey, ptxt_yini);

		if (flagSendU == 1 || ( (t+1) % Trefresh == 0 ) )
		{
			rep_uini = Fill(uini,slots);	
			ptxt_uini = cc->MakeCKKSPackedPlaintext(rep_uini);		
			ctxt_uini = cc->Encrypt(keys.publicKey, ptxt_uini);	
		}

		timeClientEnc = TOC(t1);
		cout << "Time for encoding and encrypting at the client at time " << t+1 << ": " << timeClientEnc << " ms" << endl;	

		timeStep = TOC(t2);		
		cout << "\nTotal time for evaluation at time " << t << ": " << timeStep << " ms" << endl << endl;	

	}	

	timeEval = TOC(t);
	cout << "Total time for evaluation for " << T << " steps: " << timeEval << " ms" << endl;	

	plant->printYU(); // print all inputs and outputs at the end of the simulation


}

