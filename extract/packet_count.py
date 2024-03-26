def round(x,base):
    return int(base * round(float(x) / base))

def pkt_cnt_features(size):

    #追加する特徴
    features = []

    #総パケット数
    total_cnt=0
    for _ in range(len(size)):
        total_cnt+=1
    features.append(total_cnt)

    inpkt_cnt=0
    outpkt_cnt=0
    for t_l in range(len(size)):
        #受信パケット数
        if(t_l[1] > 0):
            inpkt_cnt+=1
        #送信パケット数
        else:
            outpkt_cnt+=1
    features.append(outpkt_cnt)
    features.append(inpkt_cnt)

    #受信パケット数と総パケット数の比率
    in_ratio = inpkt_cnt / total_cnt
    features.append(in_ratio * 100)

    #送信パケット数と総パケット数の比率
    out_ratio = outpkt_cnt / total_cnt
    features.append(out_ratio * 100)

    #1~3を15刻みで四捨五入した値
    features.append(round(total_cnt,15))
    features.append(round(outpkt_cnt,15))
    features.append(round(inpkt_cnt,15))

    #4,5を5刻みで四捨五入した値
    features.append(round(in_ratio * 100 , 5))
    features.append(round(out_ratio * 100 , 5))

    #総/受信/送信パケットの合計サイズ Torは512byteで統一
    features.append(total_cnt * 512)
    features.append(inpkt_cnt * 512)
    features.append(outpkt_cnt * 512)

    return features