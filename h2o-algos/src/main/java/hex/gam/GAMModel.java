package hex.gam;

import hex.*;
import hex.deeplearning.DeepLearningModel;
import hex.gam.MatrixFrameUtils.AddGamColumns;
import hex.glm.GLM;
import hex.glm.GLMModel;
import hex.glm.GLMModel.GLMParameters.Family;
import hex.glm.GLMModel.GLMParameters.GLMType;
import hex.glm.GLMModel.GLMParameters.Link;
import hex.glm.GLMModel.GLMParameters.Solver;
import water.*;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;
import water.fvec.Vec;
import water.udf.CFuncRef;
import water.util.ArrayUtils;
import water.util.FrameUtils;
import water.util.Log;
import water.util.TwoDimTable;

import java.io.Serializable;
import java.util.Arrays;

import static hex.gam.MatrixFrameUtils.GamUtils.equalColNames;
import static hex.glm.GLMModel.GLMParameters.MissingValuesHandling;

public class GAMModel extends Model<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {
  public boolean _centerGAM = false;  // true if centering is needed for training dataset
  public String[][] _gamColNamesNoCentering; // store column names only for GAM columns
  public String[][] _gamColNames; // store column names only for GAM columns after decentering
  public Key<Frame>[] _gamFrameKeys;
  public Key<Frame>[] _gamFrameKeysCenter;
  public int _nclass; // 2 for binomial, > 2 for multinomial and ordinal
  public double[] _ymu;
  public long _nobs;
  public long _nullDOF;
  public int _rank;

  @Override public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) {
    if (domain==null && (_parms._family==Family.binomial || _parms._family==Family.quasibinomial || 
            _parms._family==Family.negativebinomial))
      domain = new String[]{"0","1"};
    GLMModel.GLMWeightsFun glmf = new GLMModel.GLMWeightsFun(_parms._family, _parms._link, _parms._tweedie_variance_power,
            _parms._tweedie_link_power, _parms._theta);
    return new MetricBuilderGAM(domain, _ymu, glmf, _rank, true, _parms._intercept, _nclass);
  }

  public GAMModel(Key<GAMModel> selfKey, GAMParameters parms, GAMModelOutput output) {
    super(selfKey, parms, output);
    assert(Arrays.equals(_key._kb, selfKey._kb));
  }

  public GAMModel(Key<GAMModel> selfKey, GAMParameters parms, GAMModelOutput output, String[][] gamColNamesNoCenter) {
    super(selfKey, parms, output);
    assert(Arrays.equals(_key._kb, selfKey._kb));
  }

  ModelMetricsSupervised makeModelMetrics(Frame origFr, Frame adaptFr, String description) {
    Log.info("Making metrics: " + description);
    ModelMetrics.MetricBuilder mb = scoreMetrics(adaptFr);
    ModelMetricsSupervised mm = (ModelMetricsSupervised) mb.makeModelMetrics(this, origFr, adaptFr, null);
    mm._description = description;
    return mm;
  }

  @SuppressWarnings("WeakerAccess")
  public static class GAMParameters extends Model.Parameters {
    // the following parameters will be passed to GLM algos
    public boolean _standardize = false; // pass to GLM algo
    public Family _family;
    public Link _link;
    public Solver _solver = Solver.AUTO;
    public double _tweedie_variance_power;
    public double _tweedie_link_power;
    public double _theta; // 1/k and is used by negative binomial distribution only
    public double _invTheta;
    public double [] _alpha;
    public double [] _lambda;
    public Serializable _missing_values_handling = MissingValuesHandling.MeanImputation;
    public double _prior = -1;
    public boolean _lambda_search = false;
    public int _nlambdas = -1;
    public boolean _non_negative = false;
    public boolean _exactLambdas = false;
    public double _lambda_min_ratio = -1; // special
    public boolean _use_all_factor_levels = false;
    public int _max_iterations = -1;
    public boolean _intercept = true;
    public double _beta_epsilon = 1e-4;
    public double _objective_epsilon = -1;
    public double _gradient_epsilon = -1;
    public double _obj_reg = -1;
    public boolean _compute_p_values = false;
    public boolean _remove_collinear_columns = false;
    public String[] _interactions=null;
    public StringPair[] _interaction_pairs=null;
    public boolean _early_stopping = true;
    public Key<Frame> _beta_constraints = null;
    public Key<Frame> _plug_values = null;
    // internal parameter, handle with care. GLM will stop when there is more than this number of active predictors (after strong rule screening)
    public int _max_active_predictors = -1;
    public boolean _stdOverride; // standardization override by beta constraints

    // the following parameters are for GAM
    public int[] _k; // array storing number of knots per basis function
    public double[][] _knots;// store knots for each gam column specified in _gam_X
    public String[] _gam_X; // array storing which predictor columns are needed
   // public BSType _bs; // choose spline function for gam column
    public int[] _bs; // choose spline function for gam column, 0 = cr
    public double[] _scale;  // array storing scaling values to control wriggliness of fit
    public GLMType _glmType = GLMType.gam; // internal parameter
    public boolean _saveZMatrix = false;  // if asserted will save Z matrix
    public boolean _saveGamCols = false;  // if true will save the keys to gam Columns only
    public boolean _savePenaltyMat = false; // if true will save penalty matrices as tripple array

    public String algoName() { return "GAM"; }
    public String fullName() { return "General Additive Model"; }
    public String javaName() { return GAMModel.class.getName(); }

    @Override
    public long progressUnits() {
      return 1;
    }

    public long _seed = -1;

    public enum BSType {
      cr  // will support more in the future
    }

    public InteractionSpec interactionSpec() {
      return InteractionSpec.create(_interactions, _interaction_pairs);
    }
    
    public MissingValuesHandling missingValuesHandling() {
      if (_missing_values_handling instanceof MissingValuesHandling)
        return (MissingValuesHandling) _missing_values_handling;
      assert _missing_values_handling instanceof DeepLearningModel.DeepLearningParameters.MissingValuesHandling;
      switch ((DeepLearningModel.DeepLearningParameters.MissingValuesHandling) _missing_values_handling) {
        case MeanImputation:
          return MissingValuesHandling.MeanImputation;
        case Skip:
          return MissingValuesHandling.Skip;
        default:
          throw new IllegalStateException("Unsupported missing values handling value: " + _missing_values_handling);
      }
    }

    public DataInfo.Imputer makeImputer() {
      if (missingValuesHandling() == MissingValuesHandling.PlugValues) {
        if (_plug_values == null || _plug_values.get() == null) {
          throw new IllegalStateException("Plug values frame needs to be specified when Missing Value Handling = PlugValues.");
        }
        return new GLM.PlugValuesImputer(_plug_values.get());
      } else { // mean/mode imputation and skip (even skip needs an imputer right now! PUBDEV-6809)
        return new DataInfo.MeanImputer();
      }
    }

    public final static double linkInv(double x, Link link, double tweedie_link_power) {
      switch(link) {
//        case multinomial: // should not be used
        case identity:
          return x;
        case ologlog:
          return 1.0-Math.exp(-1.0*Math.exp(x));
        case ologit:
        case logit:
          return 1.0 / (Math.exp(-x) + 1.0);
        case log:
          return Math.exp(x);
        case inverse:
          double xx = (x < 0) ? Math.min(-1e-5, x) : Math.max(1e-5, x);
          return 1.0 / xx;
        case tweedie:
          return tweedie_link_power == 0
                  ?Math.max(2e-16,Math.exp(x))
                  :Math.pow(x, 1/ tweedie_link_power);
        default:
          throw new RuntimeException("unexpected link function  " + link.toString());
      }
    }
  }

  public static class GAMModelOutput extends Model.Output {
    public String[] _coefficient_names_no_centering;
    public String[] _coefficient_names;    
    public TwoDimTable _glm_model_summary;
    public ModelMetrics _glm_training_metrics;
    public ModelMetrics _glm_validation_metrics;
    public TwoDimTable _model_summary_GAM;
    public ModelMetrics _glm_cross_validation_metrics;
    public double _glm_dispersion;
    public double[] _glm_zvalues;
    public double[] _glm_pvalues;
    public double[][] _glm_vcov;
    public double[] _glm_stdErr;
    public double _glm_best_lamda_value;
    public TwoDimTable _glm_scoring_history;
    public int _best_lambda_idx; // lambda which minimizes deviance on validation (if provided) or train (if not)
    public int _lambda_1se = -1; // lambda_best + sd(lambda); only applicable if running lambda search with nfold
    public int _selected_lambda_idx; // lambda which minimizes deviance on validation (if provided) or train (if not)
    public double[] _model_beta_no_centering; // coefficients generated during model training
    public double[] _standardized_model_beta_no_centering; // standardized coefficients generated during model training
    public double[] _model_beta; // coefficients generated during model training
    public double[] _standardized_model_beta; // standardized coefficients generated during model training
    public double[][] _model_beta_multinomial_no_centering;  // store multinomial coefficients during model training
    public double[][] _standardized_model_beta_multinomial_no_centering;  // store standardized multinomial coefficients during model training
    public double[][] _model_beta_multinomial;  // store multinomial coefficients during model training
    public double[][] _standardized_model_beta_multinomial;  // store standardized multinomial coefficients during model training
    private double[] _zvalues;
    private double _dispersion;
    private boolean _dispersionEstimated;
    public double[][][] _zTranspose; // Z matrix for de-centralization, can be null
    public double[][][] _penaltyMatrices_center; // stores t(Z)*t(D)*Binv*D*Z and can be null
    public double[][][] _penaltyMatrices;          // store t(D)*Binv*D and can be null
    public double[][][] _binvD; // store BinvD for each gam column specified for scoring
    public double[][] _knots; // store knots location for each gam column
    public int[] _numKnots;  // store number of knots per gam column
    public Key<Frame> _gamTransformedTrainCenter;  // contain key of predictors, all gam columns centered
    public Key<Frame> _gamGamXCenter; // store key for centered GAM columns if needed
    public DataInfo _dinfo;
    public String[] _responseDomains;

    public double dispersion(){ return _dispersion;}

    public GAMModelOutput(GAM b, Frame adaptr, DataInfo dinfo) {
      super(b, adaptr);
      _dinfo = dinfo;
      _domains = dinfo._adaptedFrame.domains(); // get domain of dataset predictors
      _responseDomains = dinfo._adaptedFrame.lastVec().domain();
    }

    @Override public ModelCategory getModelCategory() {
      return ModelCategory.Regression;
    } // will add others later
  }

  /**
   * This method will massage the input training frame such that it can be used for scoring for a GAM model.
   *
   * @param test Testing Frame, updated in-place
   * @param expensive Try hard to adapt; this might involve the creation of
   *  whole Vecs and thus get expensive.  If {@code false}, then only adapt if
   *  no warnings and errors; otherwise just the messages are produced.
   *  Created Vecs have to be deleted by the caller (e.g. Scope.enter/exit).
   * @param computeMetrics
   * @return
   */
  @Override
  public String[] adaptTestForTrain(Frame test, boolean expensive, boolean computeMetrics) {
    // compare column names with test frame.  If equal, call adaptTestForTrain.  Otherwise, need to adapt it first
    String[] testNames = test.names();
    if (!equalColNames(testNames, _output._dinfo._adaptedFrame.names(), _parms._response_column)) {  // shallow check: column number, column names only
      Frame adptedF = cleanUpInputFrame(test, testNames);
      int testNumCols = test.numCols();
      for (int index=0; index < testNumCols; index++)
        test.remove(0);
      int adaptNumCols = adptedF.numCols();
      for (int index=0; index < adaptNumCols; index++)
        test.add(adptedF.name(index), adptedF.vec(index));

      return super.adaptTestForTrain(adptedF, expensive, computeMetrics);
    }
    return super.adaptTestForTrain(test, expensive, computeMetrics);
  }

  public Frame cleanUpInputFrame(Frame test, String[] testNames) {
    Frame adptedF = new Frame(Key.make(), test._names.clone(), test.vecs().clone()); // clone test dataset
    int numGamCols = _output._numKnots.length;
    Vec[] gamCols = new Vec[numGamCols];
    for (int vind=0; vind<numGamCols; vind++)
      gamCols[vind] = adptedF.vec(_parms._gam_X[vind]).clone();
    Frame onlyGamCols = new Frame(_parms._gam_X, gamCols);
    AddGamColumns genGamCols = new AddGamColumns(_output._binvD, _output._knots, _output._numKnots, _output._dinfo,
            onlyGamCols);
    genGamCols.doAll(genGamCols._gamCols2Add, Vec.T_NUM, onlyGamCols);
    String[] gamColsNames = new String[genGamCols._gamCols2Add];
    int offset = 0;
    for (int ind=0; ind<genGamCols._numGAMcols; ind++) {
      System.arraycopy(_gamColNamesNoCentering[ind], 0, gamColsNames, offset, _gamColNamesNoCentering[ind].length);
      offset+= _gamColNamesNoCentering[ind].length;
    }
    Frame oneAugmentedColumn = genGamCols.outputFrame(Key.make(), gamColsNames, null);
    if (_parms._ignored_columns != null) {  // remove ignored columns
      for (String iname:_parms._ignored_columns) {
        if (ArrayUtils.contains(testNames, iname)) {
          adptedF.remove(iname);
        }
      }
    }
    int numCols = adptedF.numCols();  // remove constant or bad frames.
    for (int vInd=0; vInd<numCols; vInd++) {
      Vec v = adptedF.vec(vInd);
      if ((_parms._ignore_const_cols &&  v.isConst()) || v.isBad())
        adptedF.remove(vInd);
    }
    Vec respV = null;
    if (ArrayUtils.contains(testNames, _parms._response_column))
      respV = adptedF.remove(_parms._response_column);
    adptedF.add(oneAugmentedColumn.names(), oneAugmentedColumn.removeAll());
    Scope.track(oneAugmentedColumn);
    if (respV != null)
      adptedF.add(_parms._response_column, respV);
    return adptedF;
  }

  @Override
  protected Frame predictScoreImpl(Frame fr, Frame adaptFrm, String destination_key, Job j, boolean computeMetrics,
                                   CFuncRef customMetricFunc) {
    int nReponse = ArrayUtils.contains(adaptFrm.names(), _parms._response_column)?1:0;
    String[] predictNames = super.makeScoringNames();
    String[][] domains = new String[predictNames.length][];
    DataInfo datainfo = new DataInfo(adaptFrm.clone(), null, nReponse,
            _parms._use_all_factor_levels || _parms._lambda_search, DataInfo.TransformType.NONE,
            DataInfo.TransformType.NONE, _parms.missingValuesHandling() == MissingValuesHandling.Skip,
            _parms.missingValuesHandling() == MissingValuesHandling.MeanImputation ||
                    _parms.missingValuesHandling() == MissingValuesHandling.PlugValues, _parms.makeImputer(),
            false, _parms._weights_column!=null, _parms._offset_column==null,
            _parms._fold_column!=null, null);
    GAMScore gs = new GAMScore(_output._dinfo, _output._model_beta_no_centering, _output._model_beta_multinomial_no_centering, _nclass, j,
            _parms._family, _output._responseDomains, this, computeMetrics);
    gs.doAll(predictNames.length, Vec.T_NUM, datainfo._adaptedFrame);
    domains[0] = gs._predDomains;
    return gs.outputFrame(Key.make(), predictNames, domains);  // place holder
  }

  private class GAMScore extends MRTask<GAMScore> {
    private DataInfo _dinfo;
    private double[] _coeffs;
    private double[][] _coeffs_multinomial;
    private int _nclass;
    private boolean _computeMetrics;
    final Job _j;
    Family _family;
    private transient double[] _eta;  // store eta calculation
    private String[] _predDomains;
    final GAMModel _m;
    private final double _defaultThreshold;
    private int _lastClass;
    ModelMetrics.MetricBuilder _mb;
    final boolean _sparse;
    private transient double[][] _vcov;
    private transient double[] _tmp;

    private GAMScore(DataInfo dinfo, double[] coeffs, double[][] coeffs_multinomial, int nclass, Job job, Family family,
                     String[] domains, GAMModel m, boolean computeMetrics) {
      _dinfo = dinfo;
      _coeffs = coeffs;
      _coeffs_multinomial = coeffs_multinomial;
      _nclass = nclass;
      _computeMetrics = dinfo._responses > 0; // can only compute metrics if response column exists
      _j = job;
      _family = family;
      _predDomains = domains; // prediction/response domains
      _m = m;
      _defaultThreshold = m.defaultThreshold();
      _lastClass = _nclass-1;
      _computeMetrics=computeMetrics;
      _sparse = FrameUtils.sparseRatio(dinfo._adaptedFrame) < 0.5;
    }

    @Override
    public void map(Chunk[]chks, NewChunk[] nc) {
      if (isCancelled() || _j != null && _j.stop_requested()) return;
      if (_family.equals(Family.ordinal)||_family.equals(Family.multinomial))
        _eta = MemoryManager.malloc8d(_nclass);
      _vcov = _m._output._glm_vcov;
      if (_vcov != null)
        _tmp = MemoryManager.malloc8d(_vcov.length);
      int numPredVals = _nclass<=1?1:_nclass+1; // number of predictor values expected.
      double[] predictVals = MemoryManager.malloc8d(numPredVals);
      float[] trueResponse = null;
      double[] ps;
      if (_computeMetrics) {
        _mb = _m.makeMetricBuilder(_predDomains);
        trueResponse = new float[1];
      }
      DataInfo.Row r = _dinfo.newDenseRow();
      int chkLen = chks[0]._len;
      for (int rid=0; rid<chkLen; rid++) {  // extract each row
        _dinfo.extractDenseRow(chks, rid, r);
        processRow(r, predictVals, nc, numPredVals);
        if (_computeMetrics) {
          trueResponse[0] = (float) r.response[0];
          _mb.perRow(predictVals, trueResponse, r.weight, r.offset, _m);
        }
      }
      if (_j != null) _j.update(1);
    }

    private void processRow(DataInfo.Row r, double[] ps, NewChunk[] preds, int ncols) {
      if (r.predictors_bad)
        Arrays.fill(ps, Double.NaN);  // output NaN with bad predictor entries
      else if (r.weight == 0)
        Arrays.fill(ps, 0.0); // zero weight entries got 0 too
      switch (_family) {
        case multinomial: ps = scoreMultinomialRow(r, r.offset, ps); break;
        case ordinal: ps = scoreOrdinalRow(r, r.offset, ps); break;
        default: ps = scoreRow(r, r.offset, ps); break;
      }
      for (int predCol=0; predCol < ncols; predCol++) { // write prediction to NewChunk
        preds[predCol].addNum(ps[predCol]);
      }
      if (_vcov != null) 
        preds[ncols].addNum(Math.sqrt(r.innerProduct(r.mtrxMul(_vcov, _tmp))));
    }

    public double[] scoreRow(DataInfo.Row r, double offset, double[] preds) {
      double mu = _m._parms.linkInv(r.innerProduct(_coeffs) + offset, _m._parms._link,
              _m._parms._tweedie_link_power);
      if (_m._parms._family == GLMModel.GLMParameters.Family.binomial ||
              _m._parms._family == GLMModel.GLMParameters.Family.quasibinomial) { // threshold for prediction
        preds[0] = mu >= _defaultThreshold?1:0;
        preds[1] = 1.0 - mu; // class 0
        preds[2] = mu; // class 1
      } else
        preds[0] = mu;

      return preds;
    }

    public double[] scoreOrdinalRow(DataInfo.Row r, double offset, double[] preds) {
      final double[][] bm = _coeffs_multinomial;
      Arrays.fill(preds,0); // initialize to small number
      preds[0] = _lastClass;  // initialize to last class by default here
      double previousCDF = 0.0;
      for (int cInd = 0; cInd < _lastClass; cInd++) { // classify row and calculate PDF of each class
        double eta = r.innerProduct(bm[cInd]) + offset;
        double currCDF = 1.0 / (1 + Math.exp(-eta));
        preds[cInd + 1] = currCDF - previousCDF;
        previousCDF = currCDF;

        if (eta > 0) { // found the correct class
          preds[0] = cInd;
          break;
        }
      }
      for (int cInd = (int) preds[0] + 1; cInd < _lastClass; cInd++) {  // continue PDF calculation
        double currCDF = 1.0 / (1 + Math.exp(-r.innerProduct(bm[cInd]) + offset));
        preds[cInd + 1] = currCDF - previousCDF;
        previousCDF = currCDF;

      }
      preds[_nclass] = 1-previousCDF;
      return preds;
    }

    public double[] scoreMultinomialRow(DataInfo.Row r, double offset, double[] preds) {
      double[] eta = _eta;
      final double[][] bm = _coeffs_multinomial;
      double sumExp = 0;
      double maxRow = Double.NEGATIVE_INFINITY;
      for (int c = 0; c < bm.length; ++c) {
        eta[c] = r.innerProduct(bm[c]) + offset;
        if(eta[c] > maxRow)
          maxRow = eta[c];
      }
      for (int c = 0; c < bm.length; ++c)
        sumExp += eta[c] = Math.exp(eta[c]-maxRow); // intercept
      sumExp = 1.0 / sumExp;
      for (int c = 0; c < bm.length; ++c)
        preds[c + 1] = eta[c] * sumExp;
      preds[0] = ArrayUtils.maxIndex(eta);
      return preds;
    }
  }

  @Override
  public double[] score0(double[] data, double[] preds) {
    throw new UnsupportedOperationException("GAMModel.score0 should never be called");
  }

  @Override protected Futures remove_impl(Futures fs, boolean cascade) {

    if (_centerGAM) {
      Keyed.remove(_output._gamTransformedTrainCenter, fs, true);
      Keyed.remove(_output._gamGamXCenter, fs, true);
    }
    super.remove_impl(fs, cascade);
    return fs;
  }

  @Override protected AutoBuffer writeAll_impl(AutoBuffer ab) {
    if (_output._gamTransformedTrainCenter!=null)
      ab.putKey(_output._gamTransformedTrainCenter);
    return super.writeAll_impl(ab);
  }

  @Override protected Keyed readAll_impl(AutoBuffer ab, Futures fs) {
    return super.readAll_impl(ab, fs);
  }
}
