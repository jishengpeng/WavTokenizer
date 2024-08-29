# 测试各种指标
import os
import glob
from UTMOS import UTMOSScore
from periodicity import calculate_periodicity_metrics
import torchaudio
from pesq import pesq
import numpy as np
import torch
import math
from pystoi import stoi

device=torch.device('cuda:0')

# 如果是ljspeech，需要更换路径，更换数据读取逻辑，更换stoi的采样率

def main():
    prepath="./Result/Minicodec/infer/dac_nq4_all"
    rawpath="./Data/libritts/test-clean"
    # rawpath="./Data/LJSpeech-1.1/wavs"
    preaudio = os.listdir(prepath)
    rawaudio = []

    UTMOS=UTMOSScore(device='cuda:0')
    
    # libritts
    for i in range(len(preaudio)):
        id1=preaudio[i].split('_')[0]
        id2=preaudio[i].split('_')[1]
        rawaudio.append(rawpath+"/"+id1+"/"+id2+"/"+preaudio[i])

    # # ljspeech
    # for i in range(len(preaudio)):
    #     rawaudio.append(rawpath+"/"+preaudio[i])

    utmos_sumgt=0
    utmos_sumencodec=0
    pesq_sumpre=0
    f1score_sumpre=0
    stoi_sumpre=[]
    f1score_filt=0

    for i in range(len(preaudio)):
        print(i)
        rawwav,rawwav_sr=torchaudio.load(rawaudio[i])
        prewav,prewav_sr=torchaudio.load(prepath+"/"+preaudio[i])
        # breakpoint()
        rawwav=rawwav.to(device)
        prewav=prewav.to(device)
        # print(rawwav.size(),prewav.size())
        # breakpoint()
        rawwav_16k=torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=16000)  #测试UTMOS的时候必须重采样
        prewav_16k=torchaudio.functional.resample(prewav, orig_freq=prewav_sr, new_freq=16000)


        # 1.UTMOS
        print("****UTMOS_raw",i,UTMOS.score(rawwav_16k.unsqueeze(1))[0].item())
        print("****UTMOS_encodec",i,UTMOS.score(prewav_16k.unsqueeze(1))[0].item())
        utmos_sumgt+=UTMOS.score(rawwav_16k.unsqueeze(1))[0].item()
        utmos_sumencodec+=UTMOS.score(prewav_16k.unsqueeze(1))[0].item()
    

        # breakpoint()

        ## 2.PESQ  
        min_len=min(rawwav_16k.size()[1],prewav_16k.size()[1])
        rawwav_16k_pesq=rawwav_16k[:,:min_len].squeeze(0)
        prewav_16k_pesq=prewav_16k[:,:min_len].squeeze(0)
        pesq_score = pesq(16000, rawwav_16k_pesq.cpu().numpy(), prewav_16k_pesq.cpu().numpy(), "wb", on_error=1)
        print("****PESQ",i,pesq_score)
        pesq_sumpre+=pesq_score
        # breakpoint()

        ## 3.F1-score
        min_len=min(rawwav_16k.size()[1],prewav_16k.size()[1])
        rawwav_16k_f1score=rawwav_16k[:,:min_len]
        prewav_16k_f1score=prewav_16k[:,:min_len]
        periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(rawwav_16k_f1score,prewav_16k_f1score)
        print("****f1",periodicity_loss, pitch_loss, f1_score,f1score_sumpre)
        if(math.isnan(f1_score)):
            f1score_filt+=1
            print("*****",f1score_filt)
        else:
            f1score_sumpre+=f1_score
        # breakpoint()


        ## 4.STOI
        # # 针对重采样的ljspeech
        # rawwav_24k=torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=24000)
        # min_len=min(rawwav_24k.size()[1],prewav.size()[1])
        # rawwav_stoi=rawwav_24k[:,:min_len].squeeze(0)
        # prewav_stoi=prewav[:,:min_len].squeeze(0)
        # tmp_stoi=stoi(rawwav_stoi.cpu(),prewav_stoi.cpu(),24000,extended=False)
        # print("****stoi",tmp_stoi)
        # stoi_sumpre.append(tmp_stoi)
        # # breakpoint()

        # 针对libritts采样率是24k的
        min_len=min(rawwav.size()[1],prewav.size()[1])
        rawwav_stoi=rawwav[:,:min_len].squeeze(0)
        prewav_stoi=prewav[:,:min_len].squeeze(0)
        tmp_stoi=stoi(rawwav_stoi.cpu(),prewav_stoi.cpu(),rawwav_sr,extended=False)
        print("****stoi",tmp_stoi)
        stoi_sumpre.append(tmp_stoi)

    print("*************UTMOS_raw",utmos_sumgt,utmos_sumgt/len(preaudio))
    print("*************UTMOS_encodec",utmos_sumgt,utmos_sumencodec/len(preaudio))
    print("*************PESQ:",pesq_sumpre,pesq_sumpre/len(preaudio))
    print("*************F1_score:",f1score_sumpre,f1score_sumpre/(len(preaudio)-f1score_filt),f1score_filt)
    print("*************STOI:",np.mean(stoi_sumpre))
    
    

if __name__=="__main__":
    main()