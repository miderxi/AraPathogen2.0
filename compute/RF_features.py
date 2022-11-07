import numpy as np

class Features_C123:
    def __init__(self,info:list):
        class MyFeature():
            def __init__(self,pkl_files, allow_empty=False):
                self.data_loaded = False
                self.data_files = pkl_files
                self.allow_empty = allow_empty #host only feature,
                self.data = None
                self.data_shape = None

            def __getitem__(self,index):
                if not self.data_loaded:
                    self.data = dict()
                    for pkl_file in self.data_files:
                        self.data.update(np.load(pkl_file, allow_pickle=True))
                    self.data_loaded = True
                    self.data_shape = list(self.data.values())[0].shape 
                    if 'average' not in self.data.keys():
                        self.data['average'] = np.array(list(self.data.values())).mean(0)

                if not self.allow_empty:
                    v = self.data[index]
                else:
                    if index in self.data.keys():
                        v = self.data[index] 
                    else:
                        v = self.data['average']
                return v

        #feature_ac  = MyFeature(["../features/AC/ara_and_eff_AC_210.pkl"])  #shape(210,)
        #feature_ct  = MyFeature(["../features/CT/ara_and_eff_CT_343.pkl"])  #shape (343,)
        #feature_dpc = MyFeature(["../features/DPC/ara_and_eff_DPC_400.pkl"])                #shape (400,)
        #feature_PSP    = MyFeature(["../features/SecondaryStructure/features_secd.pkl"])
        #feature_AraNetProperty = MyFeature(["../features/AraNetProperty/ara_and_eff_AraNet_pure_property_C1.pkl"],allow_empty=True)
        
        feature_EsmMean   = MyFeature(["../features/esm-mean/ara_and_eff_EsmMean_v2.pkl"])  #shape(1280,)
        feature_doc2vec   = MyFeature(["../features/doc2vec/ara_and_eff_doc2vec.pkl"])  #shape (400,)
        feature_AraNetStruc2vec = MyFeature(["../features/AraNetStruc2vec/AraNetStruc2vec_256.pkl"],allow_empty=True)
        feature_AraNetNode2vec  = MyFeature(["../features/AraNetNode2vec/AraNet_node2vec_256.pkl"],allow_empty=True) #
        feature_cksaap = MyFeature(["../features/CKSAAP/ara_and_eff_CKSAAP.pkl"])      #shape (1200,)
        feature_EsmMsa    = MyFeature(["../features/esm-msa/ara_and_eff_esm-msa.pkl"])  #shape (768,)
        feature_prottrans = MyFeature(["../features/prottrans/ara_and_eff_prottrans_v2.pkl"]) #shape (1024,)
        #feature_sublocation = MyFeature(["../features/sublocation/sublocaiton.pkl"],allow_empty=True)       
        #feature_DMINode2vec = MyFeature(['../features/dmi_node2vec/ara_and_eff_dmi_node2vec.pkl'],allow_empty=True)
        #feature_GeneExpression = MyFeature(["../features/gene_expression/geo.pkl"],allow_empty=True)
        #feature_StringAraNetStruc2vec = MyFeature([
        #    "../features/AraNetStruc2vec/string_AraNetStruc2vec_128.pkl"],allow_empty=True)
        #feature_AraNetProperty = MyFeature_fold([
        #    f"../features/AraNetProperty/ara_and_eff_AraNet_property_C1_fold{i}.pkl" for i in range(5)]) 
        #feature_target = MyFeature_fold([f"../features/ImmunityProtein/target_C1_fold{i}.pkl" for i in range(5)])

        self.info = info
        self.features={
            #'ac':feature_ac,
            #'ct':feature_ct,
            #'dpc':feature_dpc,
            #'PSP':feature_PSP,
            #'AraNetProperty':feature_AraNetProperty,
            
            'doc2vec':feature_doc2vec,
            'EsmMean':feature_EsmMean,
            'AraNetNode2vec':feature_AraNetNode2vec,
            'AraNetStruc2vec':feature_AraNetStruc2vec,
            'cksaap':feature_cksaap,
            #'AraNetTarget':feature_target,
            #'stringNet':feature_StringAraNetStruc2vec
            #'DMINode2vec':feature_DMINode2vec,
            #'GeneExpression':feature_GeneExpression,
            #'sublocation':feature_sublocation,
            'EsmMsa':feature_EsmMsa,
            'prottrans':feature_prottrans,
            }
    
    def get(self,index,foldn=None):
        return np.hstack([self.features[i][index] for i in self.info])

    def __getitem__(self,index):
        return np.hstack([self.features[i][index] for i in self.info])


