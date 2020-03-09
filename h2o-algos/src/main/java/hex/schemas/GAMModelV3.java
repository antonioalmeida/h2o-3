package hex.schemas;

import hex.gam.GAMModel;
import water.api.schemas3.ModelSchemaV3;
import water.api.schemas3.TwoDimTableV3;
import water.api.schemas3.ModelOutputSchemaV3;
import water.api.API;

public class GAMModelV3 extends ModelSchemaV3<GAMModel, GAMModelV3, GAMModel.GAMParameters, GAMV3.GAMParametersV3,
        GAMModel.GAMModelOutput, GAMModelV3.GAMModelOutputV3> {
  public static final class GAMModelOutputV3 extends ModelOutputSchemaV3<GAMModel.GAMModelOutput, GAMModelOutputV3> {
    @API(help="Table of Coefficients")
    TwoDimTableV3 coefficients_table;

    @API(help="Standardized Coefficient Magnitudes")
    TwoDimTableV3 standardized_coefficient_magnitudes;

    @API(help="Dispersion parameter, only applicable to Tweedie family")
    double dispersion;
  }

  public GAMV3.GAMParametersV3 createParametersSchema() { return new GAMV3.GAMParametersV3();}
  public GAMModelOutputV3 createOutputSchema() { return new GAMModelOutputV3();}

  @Override
  public GAMModel createImpl() {
    GAMModel.GAMParameters parms = parameters.createImpl();
    return new GAMModel(model_id.key(), parms, null);
  }
}
