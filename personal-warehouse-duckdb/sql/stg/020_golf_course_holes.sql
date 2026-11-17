-- 기준 골프장 홀 정보 (홀 번호 1~18). 시트에는 없고 앱 코드 상수에 있던 값. 티 이름은 Black · Blue · White · Red
CREATE OR REPLACE TABLE stg.golf_course_holes AS
SELECT * FROM (VALUES
    ('Maple Ridge CC', 1, 4, 7, 402, 384, 362, 318),
    ('Maple Ridge CC', 2, 5, 3, 548, 526, 501, 446),
    ('Maple Ridge CC', 3, 3, 15, 192, 175, 158, 121),
    ('Maple Ridge CC', 4, 4, 1, 441, 418, 395, 342),
    ('Maple Ridge CC', 5, 4, 11, 378, 360, 341, 297),
    ('Maple Ridge CC', 6, 3, 17, 168, 155, 141, 112),
    ('Maple Ridge CC', 7, 4, 5, 425, 404, 380, 331),
    ('Maple Ridge CC', 8, 5, 9, 532, 511, 488, 430),
    ('Maple Ridge CC', 9, 4, 13, 365, 348, 330, 286),
    ('Maple Ridge CC', 10, 4, 8, 410, 391, 369, 322),
    ('Maple Ridge CC', 11, 4, 12, 372, 356, 337, 294),
    ('Maple Ridge CC', 12, 3, 16, 181, 166, 150, 118),
    ('Maple Ridge CC', 13, 5, 2, 561, 537, 512, 455),
    ('Maple Ridge CC', 14, 4, 10, 389, 371, 350, 305),
    ('Maple Ridge CC', 15, 4, 4, 436, 413, 390, 339),
    ('Maple Ridge CC', 16, 3, 18, 155, 146, 135, 104),
    ('Maple Ridge CC', 17, 5, 6, 520, 500, 478, 421),
    ('Maple Ridge CC', 18, 4, 14, 398, 380, 358, 313)
) AS t(base_course, hole_no, par, hdcp, yards_black, yards_blue, yards_white, yards_red);
