from base import db
from base.com.vo.detection_vo import AdminVO,DetectionVO


def get_admin_by_username(self, username):
    return AdminVO.query.filter_by(login_username=username).first()




class AdminDAO:
    def insert_admin(self, username, hashed_password):
        raise Exception(
            "Direct creation of admin accounts is not allowed. Use the `create_admin` function instead.")

    def get_admin_by_username(self, username):
        return AdminVO.query.filter_by(login_username=username).first()


class DetectionDAO:
    def insert_person_counts(self, detect_vo):
        db.session.add(detect_vo)
        db.session.commit()


    def get_all_counts(self):
        detection_latest = DetectionVO.query.order_by(DetectionVO.detection_id.desc()).all()

        return detection_latest
    

    def get_statistics(self, graph_path):
        graph=f"base\static\graphs/{graph_path}"
        
        detection_vo_list = DetectionVO.query.filter_by(
            graph_path=graph).all()
        return detection_vo_list
    
    

