import yaml

name_of_column = {
    'lop':['lop_cu_da_hoc'],
    'mssv':['ma_sinh_vien_cu'],
    'khoa':['khoa_tt'],
    'tai':['d_noi_tao'],
    'truong_hoc':['truong_cu_da_hoc'],
    'nganh_hoc':['nganh_cu_da_hoc', 'nganh', 'sinh_vien_nganh', 'bac_cd_nganh', 'bac_tccn_nganh'],
    'ten_mon_hoc':['ten_mon_hoc_cu', 'ten_mon_hoc_dang_hoc', 'ten_hoc_phan', 'ten_mon', 'ten_mon_ngoai', 'ten_mon_ngoai_1', 'ten_mon_ngoai_2'],
    'ma_mon_hoc':['ma_mon_hoc_cu', 'ma_mon_hoc_dang_hoc', 'ma_hoc_phan', 'ma_mh', 'ma_mh_ngoai', 'ma_mh_ngoai_1', 'ma_mh_ngoai_2'],
    'so_tin_chi':['so_tin_chi_cu', 'so_tin_chi_ngoai', 'so_tin_chi_ngoai_1', 'so_tin_chi_ngoai_2'],
    'diem':['diem_chuyen', 'diem_tich_luy_toan_khoa', 'nghe', 'noi', 'doc', 'viet', 'tong_diem'],
    'hoc_ky':['cd_tu_hoc_ki', 'tccn_tu_hoc_ki'],
    'nam_thu':['cd_nam_thu', 'tccn_nam_thu'],
    'ngay_day_du':['ngay_thi', 'ngay_cap_bang'],
    'ngay':['d_ngay', 'qd_ngay'],
    'thang':['d_thang', 'qd_thang'],
    'nam':['nam_hoc', 'khoa_hoc', 'nam_tot_nghiep', 'd_nam'],
    'so_lon':['phong_thi', 'don_so', 'tin_chi_tich_luy', 'so_hieu_bang', 'so_vao_so'],
    'so_nho':['nhom', 'tiet_thi', 'so_luong_cap'],
}

other_info = {
    'da_thi_dat_chung_chi':['TOEIC', 'IELTS'],
    'hoc_ky':['1', '2', '3'],
    'nam_thu':['1', '2', '3', '4'],
    'ly_do':['bận việc nhà', 'ngủ quên', 'nộp cho cơ quan', 'nộp cho công an địa phương'],
    'bac_dh_cd':['ĐH', 'CĐ'],
    'he_cq_vlvh':['CQ', 'VLVH'],
    'bi_buoc_thoi_hoc_theo_quyet_dinh_so':['1284A', '49'],
    'quyet_dinh':['ĐHSPKT'],
    'qd_ngay':['10', '07'],
    'qd_thang':['08', '01'],
    'qd_nam_201':['8'],
    'd_nam_201':['1','2','3','4','5','6','7','8','9','0'],
    'd_nam_20':['18','19','20','21','22','23','24'],
    'he_dao_tao':['Đại trà', 'ĐT', 'Chất lượng cao', 'CLC', 'Chất lượng cao tiếng việt', 'CLV', 'Chất lượng cao tiếng anh', 'CLA'],
    'nay_toi_lam_don_nay_xin_cap_ban_sao':['Bằng tốt nghiệp', 'Bằng cấp 3', 'Bằng anh văn'],
    'da_hoan_tat_hoc_phi_va_sach_thu_vien':['rồi', 'chưa'],
}


with open('config/name_of_column.yaml', 'w', encoding='utf-8') as outfile:
    yaml.dump(name_of_column, outfile, sort_keys=False, allow_unicode=True)

with open('data/information/other_info.yaml', 'w', encoding='utf-8') as outfile:
    yaml.dump(other_info, outfile, sort_keys=False, allow_unicode=True)
